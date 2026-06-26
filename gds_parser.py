import numpy as np
import gdstk

def read_gds_filtered(path):
    """
    Optimizes reading by only loading layers 1, 2, and 4 (datatypes 0-9).
    This reduces file read and flattening times from minutes to milliseconds.
    """
    filter_set = set()
    for l in [1, 2, 4]:
        for d in range(10):
            filter_set.add((l, d))
    try:
        return gdstk.read_gds(path, filter=filter_set)
    except Exception as e:
        print(f"[WARN] Filtered read failed, falling back to full read: {e}")
        return gdstk.read_gds(path)


def parse_gds_wafer_boundary(path, layer=2, datatype=0):
    lib = read_gds_filtered(path)
    top_cells = lib.top_level()
    if not top_cells:
        raise ValueError("No top-level cells found in GDS.")
    cell = top_cells[0]
    cell.flatten()

    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapezoid is None:
        raise ImportError("No trapezoid integration function found in NumPy.")

    best_poly = None
    best_pts = None
    best_area = 0

    for poly in cell.polygons:
        pts = poly.points
        area = float(np.abs(trapezoid(pts[:, 1], pts[:, 0])))
        if area > best_area:
            best_area = area
            best_poly = poly
            best_pts = pts

    if best_poly is None:
        raise ValueError("No polygons found in GDS file.")

    # FIT A CIRCLE to GDS outer boundary to find exact geometric center (0,0).
    # This prevents systematic offsets caused by arithmetic means of asymmetric flat vertices.
    try:
        from wafer_metrology import fit_circle_least_squares
        xc, yc, R = fit_circle_least_squares(best_pts)
    except Exception as e:
        print(f"[WARN] GDS circle fit failed, falling back to arithmetic mean: {e}")
        xc = float(np.mean(best_pts[:, 0]))
        yc = float(np.mean(best_pts[:, 1]))
        R  = float(np.mean(np.sqrt((best_pts[:, 0] - xc)**2 + (best_pts[:, 1] - yc)**2)))
        
    print(f"  [Wafer Boundary Parser] Selected Layer {best_poly.layer}, Datatype {best_poly.datatype}. Center=({xc:.1f}, {yc:.1f}), R={R:.1f} um")
    return xc, yc, R


def get_gds_overlay_polygons(path, config):
    try:
        lib = read_gds_filtered(path)
    except Exception as e:
        print(f"Error reading GDS overlay: {e}")
        return []

    top_cells = lib.top_level()
    if not top_cells:
        return []
    cell = top_cells[0]

    overlay_polygons = []
    flat_cell = cell.copy("TEMP_FLAT")
    flat_cell.flatten()
    
    best_circle_poly = None
    best_area = 0
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

    for poly in flat_cell.polygons:
        p_pts = poly.points.astype(np.float64)
        area = float(np.abs(trapezoid(p_pts[:, 1], p_pts[:, 0])))
        if area > best_area:
            best_area = area
            best_circle_poly = p_pts
                    
    if best_circle_poly is not None:
        overlay_polygons.append(best_circle_poly)

    for poly in flat_cell.polygons:
        if poly.layer == 4:
            pts = poly.points.astype(np.float64)
            min_pt, max_pt = np.min(pts, axis=0), np.max(pts, axis=0)
            w, h = max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]
            dim_max, dim_min = max(w, h), min(w, h)
            
            is_square = (280.0 <= w <= 350.0) and (280.0 <= h <= 350.0)
            is_bar = (dim_max > 2000.0) and (100.0 <= dim_min <= 500.0)
            if is_square or is_bar:
                overlay_polygons.append(pts)

    polys = [p.points.astype(np.float64) for p in flat_cell.polygons if p.layer == 4]
    if not polys:
        polys = [p.points.astype(np.float64) for p in flat_cell.polygons if p.layer == 1]

    if polys:
        gds_R = 150000.0
        if best_circle_poly is not None:
            cx = float(np.mean(best_circle_poly[:, 0]))
            cy = float(np.mean(best_circle_poly[:, 1]))
            gds_R = float(np.mean(np.sqrt((best_circle_poly[:, 0] - cx)**2 + (best_circle_poly[:, 1] - cy)**2)))
        
        cells_list = get_gds_cells_list(polys, gds_R)
        for c in cells_list:
            min_x, min_y, max_x, max_y = c["bbox"]
            box = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ], dtype=np.float64)
            overlay_polygons.append(box)

    return overlay_polygons


def parse_alignment_markers(gds_path: str) -> dict[str, list[dict]]:
    try:
        lib = read_gds_filtered(gds_path)
    except Exception as e:
        print(f"Warning: Could not read markers: {e}")
        return {"left": [], "right": []}

    top_cells = lib.top_level()
    if not top_cells:
        return {"left": [], "right": []}
    cell = top_cells[0]
    cell.flatten()

    left_markers = []
    right_markers = []

    for poly in cell.polygons:
        if poly.layer == 4:
            pts = poly.points
            min_pt, max_pt = np.min(pts, axis=0), np.max(pts, axis=0)
            w, h = max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]
            cx, cy = (min_pt[0] + max_pt[0]) / 2.0, (min_pt[1] + max_pt[1]) / 2.0
            dim_max, dim_min = max(w, h), min(w, h)

            is_square = (280.0 <= w <= 350.0) and (280.0 <= h <= 350.0)
            is_bar = (dim_max > 2000.0) and (100.0 <= dim_min <= 500.0)

            if is_square or is_bar:
                marker_info = {
                    "type": "square" if is_square else "bar",
                    "bbox": (float(min_pt[0]), float(min_pt[1]), float(max_pt[0]), float(max_pt[1])),
                    "center": (float(cx), float(cy)),
                    "polygon": pts.tolist()
                }
                if cx < 0:
                    left_markers.append(marker_info)
                else:
                    right_markers.append(marker_info)

    return {"left": left_markers, "right": right_markers}


def get_gds_cells_list(polygons, gds_R):
    raw_cells = []
    for poly in polygons:
        poly_arr = np.array(poly)
        if len(poly_arr) < 3:
            continue
        min_x, min_y = np.min(poly_arr, axis=0)
        max_x, max_y = np.max(poly_arr, axis=0)
        w, h = max_x - min_x, max_y - min_y
        
        if w > 0.8 * (2 * gds_R) or h > 0.8 * (2 * gds_R):
            continue
        if w < 3000.0 or h < 3000.0:
            continue
            
        raw_cells.append({
            "bbox": (min_x, min_y, max_x, max_y),
            "center": ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        })

    if not raw_cells:
        return []

    tol_merge = 0.05 * gds_R
    merged_cells = []
    
    for raw in raw_cells:
        cx, cy = raw["center"]
        bx_min, by_min, bx_max, by_max = raw["bbox"]
        
        merged_found = False
        for mc in merged_cells:
            mc_cx, mc_cy = mc["center"]
            dist = np.sqrt((cx - mc_cx) ** 2 + (cy - mc_cy) ** 2)
            if dist < tol_merge:
                mbx_min, mby_min, mbx_max, mby_max = mc["bbox"]
                new_bbox = (
                    min(mbx_min, bx_min),
                    min(mby_min, by_min),
                    max(mbx_max, bx_max),
                    max(mby_max, by_max)
                )
                mc["bbox"] = new_bbox
                mc["center"] = ((new_bbox[0] + new_bbox[2]) / 2.0, (new_bbox[1] + new_bbox[3]) / 2.0)
                merged_found = True
                break
                
        if not merged_found:
            merged_cells.append({
                "bbox": (bx_min, by_min, bx_max, by_max),
                "center": (cx, cy)
            })

    centers = np.array([mc["center"] for mc in merged_cells])
    if len(centers) > 1:
        diffs_x = np.diff(np.sort(centers[:, 0]))
        diffs_x = diffs_x[diffs_x > 50.0]
        tol_x = np.median(diffs_x) * 0.4 if len(diffs_x) > 0 else 200.0
        
        diffs_y = np.diff(np.sort(centers[:, 1]))
        diffs_y = diffs_y[diffs_y > 50.0]
        tol_y = np.median(diffs_y) * 0.4 if len(diffs_y) > 0 else 200.0
    else:
        tol_x, tol_y = 200.0, 200.0

    y_centers = np.sort(centers[:, 1])
    unique_y = []
    for y in y_centers:
        if not unique_y or abs(y - unique_y[-1]) > tol_y:
            unique_y.append(y)
    unique_y = sorted(unique_y, reverse=True)

    x_centers = np.sort(centers[:, 0])
    unique_x = []
    for x in x_centers:
        if not unique_x or abs(x - unique_x[-1]) > tol_x:
            unique_x.append(x)
    unique_x = sorted(unique_x)

    for mc in merged_cells:
        mc_cx, mc_cy = mc["center"]
        mc["row"] = np.argmin([abs(mc_cy - uy) for uy in unique_y]) + 1
        mc["col_global"] = np.argmin([abs(mc_cx - ux) for ux in unique_x]) + 1

        row = mc["row"]
        col_glob = mc["col_global"]

        if row in [1, 10]:
            mc["col"] = col_glob - 3
        elif row in [2, 9]:
            mc["col"] = col_glob - 2
        elif row in [3, 8]:
            mc["col"] = col_glob - 1
        else:
            mc["col"] = col_glob

    return merged_cells