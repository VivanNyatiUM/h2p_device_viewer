import math
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List, Any

NOMINAL_COORDS = {
    (1, 1): (-400.0, -1400.0),
    (1, 2): (400.0, -1400.0),
    (2, 1): (-400.0, -600.0),
    (2, 2): (400.0, -600.0),
    (3, 1): (-400.0, 600.0),
    (3, 2): (400.0, 600.0),
    (4, 1): (-400.0, 1400.0),
    (4, 2): (400.0, 1400.0),
}


def get_rotated_offset(nom_x: float, nom_y: float, theta: float, scale_x: float, scale_y: float) -> Tuple[float, float]:
    """
    Applies independent scaling to X and Y before applying coordinate rotation.
    """
    sx = nom_x * scale_x
    sy = nom_y * scale_y
    dx = sx * math.cos(theta) - sy * math.sin(theta)
    dy = sx * math.sin(theta) + sy * math.cos(theta)
    return dx, dy


def find_scribe_bar_y(crop_gray: np.ndarray) -> Optional[float]:
    """
    Estimates the vertical centerline Y of the scribe bar using 1D row intensity projections.
    """
    row_means = np.mean(crop_gray, axis=1).astype(np.float64)
    window_size = 80
    if len(row_means) < window_size:
        return None
    smoothed = np.convolve(row_means, np.ones(window_size) / window_size, mode='same')
    peak_y = int(np.argmax(smoothed))
    return float(peak_y)


def generate_fallback_circles_dual(cx: float, cy: float, scale_x: float, scale_y: float) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Generates fallback coordinates when no features are snapped using dual-axis scales.
    """
    circles = {}
    for (row, col), (nom_x, nom_y) in NOMINAL_COORDS.items():
        circles[(row, col)] = (
            cx + nom_x * scale_x,
            cy + nom_y * scale_y
        )
    return circles


def get_projected_box_fallback(cx: float, cy: float, scale_x: float, scale_y: float, theta: float) -> List[Tuple[float, float]]:
    """
    Generates a projected box using global scale and rotation priors.
    """
    half_w = 160.0
    corners = []
    for dx, dy in [(-half_w, -half_w), (half_w, -half_w), (half_w, half_w), (-half_w, half_w)]:
        rx = (dx * scale_x) * math.cos(theta) - (dy * scale_y) * math.sin(theta)
        ry = (dx * scale_x) * math.sin(theta) + (dy * scale_y) * math.cos(theta)
        corners.append((cx + rx, cy + ry))
    return corners


def fit_local_obb_from_contour(
    cnt: np.ndarray,
    scale_x: float,
    scale_y: float,
    global_theta: float,
    offset_x: float,
    offset_y: float
) -> Tuple[List[Tuple[float, float]], float, float, float]:
    """
    Fits an oriented bounding box (OBB) to a local contour using cv2.minAreaRect.
    Standardizes dimensions, handles nested inner perimeters, and regularizes scale.
    """
    rect = cv2.minAreaRect(cnt)
    (cx_r, cy_r), (w_r, h_r), angle_deg = rect
    
    cx = offset_x + cx_r
    cy = offset_y + cy_r
    
    theta_local = math.radians(angle_deg)
    
    # Resolve the 90-degree alignment degeneracy of the square
    best_diff = float('inf')
    best_theta = global_theta
    best_w, best_h = w_r, h_r
    
    for k in range(-2, 3):
        candidate_theta = theta_local + k * (math.pi / 2.0)
        diff = abs(candidate_theta - global_theta)
        if diff < best_diff:
            best_diff = diff
            best_theta = candidate_theta
            if abs(k) % 2 == 1:
                best_w, best_h = h_r, w_r
            else:
                best_w, best_h = w_r, h_r
                
    # Nested inner perimeter scaling (Rows 1 and 4 have ~150px inner, ~320px outer)
    expected_outer_w = 320.0 * scale_x
    expected_outer_h = 320.0 * scale_y
    
    if best_w < 220.0 * scale_x:
        w_final = best_w * (320.0 / 150.0)
        h_final = best_h * (320.0 / 150.0)
    else:
        w_final = best_w
        h_final = best_h
        
    # Regularize to keep dimensions from jumping due to noise
    w_final = 0.85 * w_final + 0.15 * expected_outer_w
    h_final = 0.85 * h_final + 0.15 * expected_outer_h
    
    half_w = w_final / 2.0
    half_h = h_final / 2.0
    
    corners = []
    for dx, dy in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]:
        rx = dx * math.cos(best_theta) - dy * math.sin(best_theta)
        ry = dx * math.sin(best_theta) + dy * math.cos(best_theta)
        corners.append((cx + rx, cy + ry))
        
    return corners, best_theta, cx, cy


def match_column_squares_with_prior(
    crop_gray: np.ndarray,
    calibrated_theta: float,
    scale_x: float,
    scale_y: float,
    search_x_min: int,
    search_x_max: int
) -> Tuple[bool, float, float, float, float, Dict[Tuple[int, int], Tuple[float, float]], Dict[Tuple[int, int], List[Tuple[float, float]]], float]:
    """
    Scans the crop area to detect candidate square features, matches them against
    both Column-1 and Column-2 rigid templates, snaps to the winning anchor column,
    and performs a robust local scale and rotation adaptation.
    """
    h, w = crop_gray.shape
    strip = crop_gray[:, search_x_min:search_x_max]
    
    # 1. Multi-strategy local binarization with local adaptive thresholding
    strip_bg = float(np.percentile(strip, 30))
    _, t_otsu = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, t_med = cv2.threshold(strip, int(strip_bg + 25), 255, cv2.THRESH_BINARY)
    t_adapt = cv2.adaptiveThreshold(
        strip, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 101, -10
    )
    
    raw_candidates = []
    target_sq_w_outer = 320.0 * scale_x
    target_area_outer = target_sq_w_outer * target_sq_w_outer
    
    for thresh in [t_otsu, t_med, t_adapt]:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx_c, cy_c, cw_c, ch_c = cv2.boundingRect(cnt)
            gx_c = search_x_min + cx_c
            gy_c = cy_c
            area_c = cv2.contourArea(cnt)
            
            aspect_c = cw_c / ch_c if ch_c > 0 else 0.0
            if not (0.60 <= aspect_c <= 1.65):
                continue
            if not (0.25 * target_area_outer <= area_c <= 1.85 * target_area_outer):
                continue
                
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity_c = area_c / hull_area if hull_area > 0 else 0.0
            if solidity_c < 0.70:
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                ccx = search_x_min + M["m10"] / M["m00"]
                ccy = M["m01"] / M["m00"]
            else:
                ccx = gx_c + cw_c / 2.0
                ccy = gy_c + ch_c / 2.0
                
            raw_candidates.append({
                "cx": ccx, "cy": ccy,
                "x1": gx_c, "y1": gy_c,
                "x2": gx_c + cw_c, "y2": gy_c + ch_c,
                "area": area_c,
                "w": cw_c,
                "cnt": cnt
            })

    # Deduplicate spatial candidate detections
    candidates = []
    for cand in raw_candidates:
        is_dup = False
        for existing in candidates:
            if math.hypot(cand["cx"] - existing["cx"], cand["cy"] - existing["cy"]) < 20.0:
                is_dup = True
                break
        if not is_dup:
            candidates.append(cand)

    best_score = -1
    best_anchor_col = 1
    best_mapping = {}
    
    # 2. Symmetric voting loop matching candidates against both Column 1 and Column 2 templates
    for col_idx in [1, 2]:
        nom_x = -400.0 if col_idx == 1 else 400.0
        for cand in candidates:
            for r_anchor in [1, 2, 3, 4]:
                adx, ady = get_rotated_offset(nom_x, NOMINAL_COORDS[(r_anchor, col_idx)][1], calibrated_theta, scale_x, scale_y)
                cx_est = cand["cx"] - adx
                cy_est = cand["cy"] - ady
                
                temp_mapping = {r_anchor: cand}
                score = 1
                for r_other in [1, 2, 3, 4]:
                    if r_other == r_anchor:
                        continue
                    odx, ody = get_rotated_offset(nom_x, NOMINAL_COORDS[(r_other, col_idx)][1], calibrated_theta, scale_x, scale_y)
                    exp_x = cx_est + odx
                    exp_y = cy_est + ody
                    
                    best_match = None
                    min_dist = float("inf")
                    for other_cand in candidates:
                        dist = math.hypot(other_cand["cx"] - exp_x, other_cand["cy"] - exp_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = other_cand
                            
                    if min_dist < 120.0 * scale_x:
                        temp_mapping[r_other] = best_match
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_anchor_col = col_idx
                    best_mapping = temp_mapping

    # 3. Resolve coordinates from matched anchor points
    centroid_x_estimates = []
    centroid_y_estimates = []
    refined_circles = {}
    detected_contours = {}
    
    if best_score >= 2:
        success = True
        nom_x_anchor = -400.0 if best_anchor_col == 1 else 400.0
        
        for r, cand in best_mapping.items():
            adx, ady = get_rotated_offset(nom_x_anchor, NOMINAL_COORDS[(r, best_anchor_col)][1], calibrated_theta, scale_x, scale_y)
            centroid_x_estimates.append(cand["cx"] - adx)
            centroid_y_estimates.append(cand["cy"] - ady)
            refined_circles[(r, best_anchor_col)] = (cand["cx"], cand["cy"])
            detected_contours[(r, best_anchor_col)] = cand["cnt"]
            
        center_x_refined = float(np.mean(centroid_x_estimates))
        center_y_refined = float(np.mean(centroid_y_estimates))
        
        # Symmetrically preserve valid secondary column features
        other_col = 2 if best_anchor_col == 1 else 1
        nom_x_other = -400.0 if other_col == 1 else 400.0
        for r in [1, 2, 3, 4]:
            odx, ody = get_rotated_offset(nom_x_other, NOMINAL_COORDS[(r, other_col)][1], calibrated_theta, scale_x, scale_y)
            exp_x = center_x_refined + odx
            exp_y = center_y_refined + ody
            
            best_match = None
            min_dist = float("inf")
            for cand in candidates:
                dist = math.hypot(cand["cx"] - exp_x, cand["cy"] - exp_y)
                if dist < min_dist:
                    min_dist = dist
                    best_match = cand
                    
            if min_dist < 120.0 * scale_x:
                expected_w = 320.0 * scale_x
                if 0.50 * expected_w <= best_match["w"] <= 1.50 * expected_w:
                    refined_circles[(r, other_col)] = (best_match["cx"], best_match["cy"])
                    detected_contours[(r, other_col)] = best_match["cnt"]

        # Local Scale and Rotation Adaptation:
        col1_pts = [r for r in [1, 2, 3, 4] if (r, 1) in refined_circles]
        col2_pts = [r for r in [1, 2, 3, 4] if (r, 2) in refined_circles]
        
        local_scale_x = scale_x
        local_scale_y = scale_y
        local_theta = calibrated_theta
        
        # Fit scale_y regression
        y_regression_data = []
        for col in [1, 2]:
            col_rows = col1_pts if col == 1 else col2_pts
            if len(col_rows) >= 2:
                for r in col_rows:
                    nom_y = NOMINAL_COORDS[(r, col)][1]
                    gy = refined_circles[(r, col)][1]
                    y_regression_data.append((nom_y, gy))
        if len(y_regression_data) >= 2:
            try:
                A = np.vstack([np.array([p[0] for p in y_regression_data]), np.ones(len(y_regression_data))]).T
                y_vals = np.array([p[1] for p in y_regression_data])
                solved_scale_y, _ = np.linalg.lstsq(A, y_vals, rcond=None)[0]
                if 0.5 < solved_scale_y < 2.0:
                    local_scale_y = float(solved_scale_y)
                    local_scale_x = local_scale_y
            except Exception as e:
                print(f"[WARN] Local scale_y adaptation failed: {e}")

        # Calculate local horizontal spacing if we have matched pairs on both columns
        common_rows = set(col1_pts).intersection(set(col2_pts))
        if common_rows:
            pair_slopes = []
            pair_spacings = []
            for r in common_rows:
                pt_l = refined_circles[(r, 1)]
                pt_r = refined_circles[(r, 2)]
                dy = pt_r[1] - pt_l[1]
                dx = pt_r[0] - pt_l[0]
                if dx != 0:
                    pair_slopes.append(dy / dx)
                    dist = dx * math.cos(calibrated_theta) + dy * math.sin(calibrated_theta)
                    pair_spacings.append(dist)
            if pair_slopes:
                mean_slope = float(np.mean(pair_slopes))
                if abs(mean_slope) < 0.2:
                    local_theta = math.atan(mean_slope)
            if pair_spacings:
                mean_spacing = float(np.mean(pair_spacings))
                if mean_spacing > 100.0:
                    local_scale_x = mean_spacing / 800.0
        else:
            # Symmetrically solve local_theta from anchor column drift if secondary column is missing
            anchor_col_rows = col1_pts if best_anchor_col == 1 else col2_pts
            if len(anchor_col_rows) >= 2:
                try:
                    y_coords = np.array([refined_circles[(r, best_anchor_col)][1] for r in anchor_col_rows])
                    x_coords = np.array([refined_circles[(r, best_anchor_col)][0] for r in anchor_col_rows])
                    slope, _ = np.polyfit(y_coords, x_coords, 1)
                    if abs(slope) < 0.2:
                        local_theta = float(-slope)
                except:
                    pass

        # Recalculate local sub-pixel centroid estimates with adapted scales
        centroid_x_estimates = []
        centroid_y_estimates = []
        for (row, col), pt in refined_circles.items():
            nom_x, nom_y = NOMINAL_COORDS[(row, col)]
            pdx, pdy = get_rotated_offset(nom_x, nom_y, local_theta, local_scale_x, local_scale_y)
            centroid_x_estimates.append(pt[0] - pdx)
            centroid_y_estimates.append(pt[1] - pdy)
            
        center_x_refined = float(np.mean(centroid_x_estimates))
        center_y_refined = float(np.mean(centroid_y_estimates))
        
        scale_x = local_scale_x
        scale_y = local_scale_y
        calibrated_theta = local_theta
    else:
        success = False
        center_x_refined = float(search_x_min + (search_x_max - search_x_min) / 2.0)
        scribe_y = find_scribe_bar_y(crop_gray)
        center_y_refined = scribe_y if scribe_y is not None else float(h / 2.0)

    # 4. Project geometries and apply oriented rotated coordinates matching physical tilt
    final_circles = {}
    final_boxes = {}
    for (row, col), (nom_x, nom_y) in NOMINAL_COORDS.items():
        pdx, pdy = get_rotated_offset(nom_x, nom_y, calibrated_theta, scale_x, scale_y)
        px = center_x_refined + pdx
        py = center_y_refined + pdy
        
        if (row, col) in refined_circles:
            cx, cy = refined_circles[(row, col)]
        else:
            cx, cy = px, py
            
        final_circles[(row, col)] = (cx, cy)
        
        if (row, col) in detected_contours:
            cnt = detected_contours[(row, col)]
            try:
                # offset_x=search_x_min because candidates were found on strip (offset by search_x_min)
                corners, t_loc, cx_loc, cy_loc = fit_local_obb_from_contour(
                    cnt, scale_x, scale_y, calibrated_theta, search_x_min, 0
                )
                final_boxes[(row, col)] = corners
                final_circles[(row, col)] = (cx_loc, cy_loc)
            except Exception as e:
                final_boxes[(row, col)] = get_projected_box_fallback(cx, cy, scale_x, scale_y, calibrated_theta)
        else:
            final_boxes[(row, col)] = get_projected_box_fallback(cx, cy, scale_x, scale_y, calibrated_theta)
            
    return success, center_x_refined, center_y_refined, scale_x, scale_y, final_circles, final_boxes, calibrated_theta


def process_marker_detection(
    crop_gray: np.ndarray,
    click_local_x: float,
    click_local_y: float,
    expected_spacing: Optional[float] = None,
    calibrated_theta: Optional[float] = None,
    calibrated_scale_x: Optional[float] = None,
    calibrated_scale_y: Optional[float] = None
) -> Tuple[bool, float, float, float, float, Optional[Dict], Optional[Dict], float]:
    """
    Directs marker snap calculations based on calibration availability.
    """
    # PATH A: Prior-Guided Calibration Snapping
    if calibrated_theta is not None and calibrated_scale_x is not None and calibrated_scale_y is not None:
        try:
            # Scan crop width symmetrically (avoiding edge artifacts)
            search_x_min = 100
            search_x_max = 1900
            success, cx, cy, act_scale_x, act_scale_y, circles, boxes, act_theta = match_column_squares_with_prior(
                crop_gray, calibrated_theta, calibrated_scale_x, calibrated_scale_y, search_x_min, search_x_max
            )
            if success:
                return True, cx, cy, act_scale_x, act_scale_y, circles, boxes, act_theta
        except Exception as e:
            print(f"[WARN] Prior-guided match exception: {e}")

    # PATH B: Regular Detection Passes (Standard Configurations)
    configs = [
        (15.0, 30.0, 250, 0.4, 2.5, 0.45, 0.8, 800.0),   # 1. Standard
        (10.0, 20.0, 150, 0.3, 3.0, 0.35, 0.9, 900.0),   # 2. Low contrast
        (20.0, 45.0, 300, 0.4, 2.5, 0.50, 0.85, 800.0),  # 3. High noise
        (8.0,  15.0, 100, 0.25, 3.5, 0.30, 1.0, 1000.0), # 4. Weak features
        (25.0, 60.0, 400, 0.5, 2.2, 0.55, 0.75, 800.0),  # 5. Very strong contrast
    ]

    for i, config in enumerate(configs):
        success, cx, cy, spacing, circles, boxes, theta = _try_find_centroid(
            crop_gray, click_local_x, click_local_y, expected_spacing, *config
        )
        if success:
            print(f"[INFO] Centroid successfully resolved on configuration pass {i+1}.")
            # Symmetrically fit independent horizontal and vertical scale parameters on Left Marker
            y_data = []
            x_data = []
            for (row, col), (gx, gy) in circles.items():
                if (row, col) in boxes:
                    nom_x, nom_y = NOMINAL_COORDS[(row, col)]
                    y_data.append((nom_y, gy))
                    x_data.append((nom_x, gx))
            
            scale_x = spacing / 800.0
            scale_y = spacing / 800.0
            
            if len(y_data) >= 2:
                try:
                    A = np.vstack([np.array([p[0] for p in y_data]), np.ones(len(y_data))]).T
                    y_vals = np.array([p[1] for p in y_data])
                    solved_scale_y, _ = np.linalg.lstsq(A, y_vals, rcond=None)[0]
                    if 0.5 < solved_scale_y < 2.0:
                        scale_y = float(solved_scale_y)
                except Exception as e:
                    print(f"[WARN] scale_y least-squares solver failed: {e}")
                    
            if len(x_data) >= 2:
                try:
                    A = np.vstack([np.array([p[0] for p in x_data]), np.ones(len(x_data))]).T
                    x_vals = np.array([p[1] for p in x_data])
                    solved_scale_x, _ = np.linalg.lstsq(A, x_vals, rcond=None)[0]
                    if 0.5 < solved_scale_x < 2.0:
                        scale_x = float(solved_scale_x)
                except Exception as e:
                    print(f"[WARN] scale_x least-squares solver failed: {e}")
                    
            return True, cx, cy, scale_x, scale_y, circles, boxes, theta

    return False, 0.0, 0.0, 1.0, 1.0, None, None, 0.0


def _try_find_centroid(
    crop_gray: np.ndarray,
    click_local_x: float,
    click_local_y: float,
    expected_spacing: Optional[float],
    bar_offset: float,
    sq_offset: float,
    min_area: float,
    min_aspect: float,
    max_aspect: float,
    min_fill: float,
    group_tolerance: float,
    max_dist_from_click: float,
) -> Tuple[bool, float, float, float, Optional[Dict], Optional[Dict], float]:
    
    h, w = crop_gray.shape
    row_means = np.mean(crop_gray, axis=1).astype(np.float64)
    bg_level = float(np.percentile(row_means, 30))
    bar_thresh = bg_level + bar_offset

    bar_rows = np.where(row_means > bar_thresh)[0]
    if len(bar_rows) < 2:
        return False, 0.0, 0.0, 0.0, None, None, 0.0

    diffs = np.diff(bar_rows)
    breaks = np.where(diffs > 10)[0]

    segments = []
    seg_start = int(bar_rows[0])
    for b in breaks:
        segments.append((seg_start, int(bar_rows[b])))
        seg_start = int(bar_rows[b + 1])
    segments.append((seg_start, int(bar_rows[-1])))

    bar_start, bar_end = max(segments, key=lambda s: s[1] - s[0])
    bar_y = (bar_start + bar_end) / 2.0

    sq_thresh = bg_level + sq_offset
    all_squares = []

    for region, y_offset in (
        (crop_gray[:bar_start, :], 0),
        (crop_gray[bar_end + 1:, :], bar_end + 1),
    ):
        if region.shape[0] < 5:
            continue

        bin_img = (region > sq_thresh).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = float(cv2.contourArea(cnt))

            if area < min_area:
                continue
            if area > 0.15 * w * region.shape[0]:
                continue

            aspect = cw / ch if ch > 0 else 0.0
            if not (min_aspect <= aspect <= max_aspect):
                continue

            fill = area / (cw * ch) if cw * ch > 0 else 0.0
            if fill < min_fill:
                continue

            cx = x + cw / 2.0
            cy = y_offset + y + ch / 2.0
            all_squares.append({
                "cx": cx, "cy": cy,
                "w": cw, "h": ch,
                "area": area,
            })

    if len(all_squares) < 2:
        return False, 0.0, 0.0, 0.0, None, None, 0.0

    areas = np.array([s["area"] for s in all_squares])
    widths = np.array([s["w"] for s in all_squares])
    med_a = float(np.median(areas))
    med_w = float(np.median(widths))

    all_squares = [
        s for s in all_squares
        if (0.30 * med_a <= s["area"] <= 3.0 * med_a)
        and (0.40 * med_w <= s["w"] <= 2.2 * med_w)
    ]

    if len(all_squares) < 2:
        return False, 0.0, 0.0, 0.0, None, None, 0.0

    all_squares.sort(key=lambda s: s["cx"])
    med_w = float(np.median([s["w"] for s in all_squares]))

    columns = []
    for sq in all_squares:
        placed = False
        for col in columns:
            col_x = float(np.mean([s["cx"] for s in col]))
            if abs(sq["cx"] - col_x) < med_w * group_tolerance:
                col.append(sq)
                placed = True
                break
        if not placed:
            columns.append([sq])

    if len(columns) < 2:
        return False, 0.0, 0.0, 0.0, None, None, 0.0

    best_pair = None
    min_dist = float("inf")
    resolved_spacing = 0.0

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            cx_i = float(np.mean([s["cx"] for s in columns[i]]))
            cx_j = float(np.mean([s["cx"] for s in columns[j]]))
            spacing = abs(cx_i - cx_j)
            
            if expected_spacing is not None:
                if not (0.80 * expected_spacing <= spacing <= 1.20 * expected_spacing):
                    continue

            mid_x = (cx_i + cx_j) / 2.0
            dist = abs(mid_x - click_local_x)
            if dist < min_dist:
                min_dist = dist
                best_pair = (columns[i], columns[j])
                resolved_spacing = spacing

    if best_pair is None:
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                cx_i = float(np.mean([s["cx"] for s in columns[i]]))
                cx_j = float(np.mean([s["cx"] for s in columns[j]]))
                mid_x = (cx_i + cx_j) / 2.0
                dist = abs(mid_x - click_local_x)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (columns[i], columns[j])
                    resolved_spacing = abs(cx_i - cx_j)

    if best_pair is None:
        return False, 0.0, 0.0, 0.0, None, None, 0.0

    col_l_x = float(np.mean([s["cx"] for s in best_pair[0]]))
    col_r_x = float(np.mean([s["cx"] for s in best_pair[1]]))
    if col_l_x > col_r_x:
        col_l_x, col_r_x = col_r_x, col_l_x

    center_x = (col_l_x + col_r_x) / 2.0
    center_y = bar_y
    spacing_local = col_r_x - col_l_x
    margin = int(spacing_local * 0.15)
    inner_x1 = int(col_l_x + margin)
    inner_x2 = int(col_r_x - margin)

    if inner_x2 > inner_x1:
        clean_slice = crop_gray[:, inner_x1:inner_x2]
        clean_row_means = np.mean(clean_slice, axis=1).astype(np.float64)
        clean_bg = float(np.percentile(clean_row_means, 30))
        clean_thresh = clean_bg + (bar_offset * 1.2)
        
        bar_indices = np.where(clean_row_means > clean_thresh)[0]
        if len(bar_indices) >= 2:
            local_diffs = np.diff(bar_indices)
            local_breaks = np.where(local_diffs > 10)[0]
            
            local_segs = []
            local_seg_start = int(bar_indices[0])
            for b in local_breaks:
                local_segs.append((local_seg_start, int(bar_indices[b])))
                local_seg_start = int(bar_indices[b + 1])
            local_segs.append((local_seg_start, int(bar_indices[-1])))
            
            best_seg = min(local_segs, key=lambda s: abs((s[0] + s[1]) / 2.0 - bar_y))
            center_y = (best_seg[0] + best_seg[1]) / 2.0

    if math.hypot(center_x - click_local_x, center_y - click_local_y) > max_dist_from_click:
        return False, 0.0, 0.0, 0.0, None, None, 0.0

    scale_gds_to_px = resolved_spacing / 800.0
    circles_local = {}

    for (row, col), (nom_x, nom_y) in NOMINAL_COORDS.items():
        expected_x = center_x + nom_x * scale_gds_to_px
        expected_y = center_y + nom_y * scale_gds_to_px
        active_col_squares = best_pair[0] if col == 1 else best_pair[1]
        
        best_match = None
        min_dist = float("inf")
        for sq in active_col_squares:
            dist = math.hypot(sq["cx"] - expected_x, sq["cy"] - expected_y)
            if dist < min_dist and dist < 200.0 * scale_gds_to_px:
                min_dist = dist
                best_match = (sq["cx"], sq["cy"])
        
        if best_match is not None:
            circles_local[(row, col)] = best_match
        else:
            circles_local[(row, col)] = (expected_x, expected_y)

    nom_sq_size = int(med_w) if 'med_w' in locals() and med_w > 0 else 150
    refined_circles = {}
    detected_contours = {}

    for (row, col), (nom_x, nom_y) in NOMINAL_COORDS.items():
        expected_local_x, expected_local_y = circles_local[(row, col)]
        half_w = int(nom_sq_size * 1.2)
        lx1 = max(0, int(expected_local_x - half_w))
        ly1 = max(0, int(expected_local_y - half_w))
        lx2 = min(w, int(expected_local_x + half_w))
        ly2 = min(h, int(expected_local_y + half_w))
        
        local_patch = crop_gray[ly1:ly2, lx1:lx2]
        if local_patch.size > 0:
            bin_methods = []
            try:
                _, t_otsu = cv2.threshold(local_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                bin_methods.append(t_otsu)
            except:
                pass
            try:
                med_val = np.median(local_patch)
                _, t_med = cv2.threshold(local_patch, int(med_val + 20), 255, cv2.THRESH_BINARY)
                bin_methods.append(t_med)
            except:
                pass
            try:
                t_adapt = cv2.adaptiveThreshold(
                    local_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 51, -10
                )
                bin_methods.append(t_adapt)
            except:
                pass

            best_cnt = None
            min_dist_to_center = float("inf")
            patch_center_x = (lx2 - lx1) / 2.0
            patch_center_y = (ly2 - ly1) / 2.0
            
            for thresh in bin_methods:
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cx_c, cy_c, cw_c, ch_c = cv2.boundingRect(cnt)
                    area_c = cv2.contourArea(cnt)
                    
                    aspect_c = cw_c / ch_c if ch_c > 0 else 0.0
                    if not (0.70 <= aspect_c <= 1.45):
                        continue
                        
                    nom_area = nom_sq_size * nom_sq_size
                    if not (0.45 * nom_area <= area_c <= 1.55 * nom_area):
                        continue
                        
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity_c = area_c / hull_area if hull_area > 0 else 0.0
                    if solidity_c < 0.85:
                        continue
                        
                    extent_c = area_c / (cw_c * ch_c) if (cw_c * ch_c) > 0 else 0.0
                    if extent_c < 0.65:
                        continue
                        
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        ccx_c = M["m10"] / M["m00"]
                        ccy_c = M["m01"] / M["m00"]
                    else:
                        ccx_c = cx_c + cw_c / 2.0
                        ccy_c = cy_c + ch_c / 2.0
                        
                    dist_to_expected = math.hypot((lx1 + ccx_c) - expected_local_x, (ly1 + ccy_c) - expected_local_y)
                    if dist_to_expected > 1.2 * nom_sq_size:
                        continue
                        
                    dist_to_patch_center = math.hypot(ccx_c - patch_center_x, ccy_c - patch_center_y)
                    if dist_to_patch_center < min_dist_to_center:
                        min_dist_to_center = dist_to_patch_center
                        best_cnt = (cnt, ccx_c, ccy_c, cx_c, cy_c, cw_c, ch_c)
                        
                if best_cnt is not None:
                    break
                    
            if best_cnt is not None:
                cnt, ccx_c, ccy_c, cx_c, cy_c, cw_c, ch_c = best_cnt
                refined_x = lx1 + ccx_c
                refined_y = ly1 + ccy_c
                refined_circles[(row, col)] = (refined_x, refined_y)
                detected_contours[(row, col)] = (cnt, lx1, ly1)
            else:
                refined_circles[(row, col)] = (expected_local_x, expected_local_y)
        else:
            refined_circles[(row, col)] = (expected_local_x, expected_local_y)

    slopes = []
    for r in [1, 2, 3, 4]:
        pt_l = refined_circles.get((r, 1))
        pt_r = refined_circles.get((r, 2))
        if pt_l is not None and pt_r is not None:
            dy = pt_r[1] - pt_l[1]
            dx = pt_r[0] - pt_l[0]
            if dx != 0:
                slopes.append(dy / dx)
    theta = np.mean(slopes) if slopes else 0.0
    
    spacings = []
    for r in [1, 2, 3, 4]:
        pt_l = refined_circles.get((r, 1))
        pt_r = refined_circles.get((r, 2))
        if pt_l is not None and pt_r is not None:
            dist = (pt_r[0] - pt_l[0]) * math.cos(theta) + (pt_r[1] - pt_l[1]) * math.sin(theta)
            spacings.append(dist)
    refined_spacing = np.mean(spacings) if spacings else resolved_spacing
    scale_gds_to_px_refined = refined_spacing / 800.0
    
    centroid_x_estimates = []
    centroid_y_estimates = []
    for (row, col) in refined_circles.keys():
        nom_x, nom_y = NOMINAL_COORDS[(row, col)]
        pt = refined_circles[(row, col)]
        nom_x_rot = (nom_x * math.cos(theta) - nom_y * math.sin(theta)) * scale_gds_to_px_refined
        nom_y_rot = (nom_x * math.sin(theta) + nom_y * math.cos(theta)) * scale_gds_to_px_refined
        centroid_x_estimates.append(pt[0] - nom_x_rot)
        centroid_y_estimates.append(pt[1] - nom_y_rot)
            
    center_x_refined = np.mean(centroid_x_estimates) if centroid_x_estimates else center_x
    center_y_refined = np.mean(centroid_y_estimates) if centroid_y_estimates else center_y
    
    final_circles = {}
    final_boxes = {}
    for (row, col), (nom_x, nom_y) in NOMINAL_COORDS.items():
        pt = refined_circles.get((row, col))
        if pt == (center_x + nom_x * scale_gds_to_px, center_y + nom_y * scale_gds_to_px):
            nom_x_rot = (nom_x * math.cos(theta) - nom_y * math.sin(theta)) * scale_gds_to_px_refined
            nom_y_rot = (nom_x * math.sin(theta) + nom_y * math.cos(theta)) * scale_gds_to_px_refined
            cx, cy = center_x_refined + nom_x_rot, center_y_refined + nom_y_rot
            final_circles[(row, col)] = (cx, cy)
        else:
            cx, cy = pt
            final_circles[(row, col)] = (cx, cy)
            
        # Draw and output corner vertices using Oriented Bounding Boxes (OBB)
        if (row, col) in detected_contours:
            cnt, lx1, ly1 = detected_contours[(row, col)]
            try:
                corners, t_loc, cx_loc, cy_loc = fit_local_obb_from_contour(
                    cnt, scale_gds_to_px_refined, scale_gds_to_px_refined, theta, lx1, ly1
                )
                final_boxes[(row, col)] = corners
                final_circles[(row, col)] = (cx_loc, cy_loc)
            except:
                final_boxes[(row, col)] = get_projected_box_fallback(cx, cy, scale_gds_to_px_refined, scale_gds_to_px_refined, theta)
        else:
            final_boxes[(row, col)] = get_projected_box_fallback(cx, cy, scale_gds_to_px_refined, scale_gds_to_px_refined, theta)
            
    print(f"[REFINED] Resolved Sub-pixel Centroid: X={center_x_refined:.3f}, Y={center_y_refined:.3f}, spacing={refined_spacing:.3f}, tilt={theta*180/math.pi:.3f} deg")

    return True, center_x_refined, center_y_refined, refined_spacing, final_circles, final_boxes, theta