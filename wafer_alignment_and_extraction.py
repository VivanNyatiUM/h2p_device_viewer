#!/usr/bin/env python3
"""
wafer_alignment_and_extraction.py
=================================
A completely unified, standalone wafer metrology, GDS cell extraction, 
and interactive defect-labeling tool. This script integrates the functionality 
of all former modules (io_handler, alignment, gds_parser, manual_align_gui, 
stitcher, transformer) into a single execution workflow.
"""

from __future__ import annotations
import os
import sys
import re
import json
import copy
import math
import argparse
from pathlib import Path

import numpy as np
import cv2
import gdstk
from PIL import Image, ImageTk

# Tkinter imports for interactive GUI components
import tkinter as tk
from tkinter import ttk


# ===========================================================================
# 1. IO HANDLER CORE FUNCTIONS (formerly io_handler.py)
# ===========================================================================

def load_config(config_path="config.json"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r") as f:
        return json.load(f)


def load_defect_json(json_path):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Defect JSON file not found at: {json_path}")
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved transformed data to: {output_path}")


def detect_grid_size(tile_folder):
    """Scans folder for 'tile_x{col}_y{row}' patterns to find grid boundaries."""
    folder = Path(tile_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Tile folder does not exist at: {tile_folder}")
        
    pattern = re.compile(r'tile_x(\d+)_y(\d+)')
    max_col = 0
    max_row = 0
    
    for file in folder.iterdir():
        if file.is_file():
            match = pattern.search(file.name)
            if match:
                col = int(match.group(1))
                row = int(match.group(2))
                if col > max_col:
                    max_col = col
                if row > max_row:
                    max_row = row
                    
    if max_col == 0 or max_row == 0:
        raise ValueError(f"No valid tile files matching pattern 'tile_x*_y*' found in folder: {tile_folder}")
        
    return max_col, max_row


def load_exclusions(exclusions_path):
    """Loads manually excluded tile filenames as a set."""
    path = Path(exclusions_path)
    if not path.exists():
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return set(data)
    except Exception as e:
        print(f"Warning: Failed to load manual exclusions from {exclusions_path}: {e}")
        return set()


def save_exclusions(exclusions_set, exclusions_path):
    """Saves manually excluded tile filenames to a JSON array."""
    path = Path(exclusions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(sorted(list(exclusions_set)), f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save manual exclusions: {e}")


# ===========================================================================
# 2. CORE CIRCLE GEOMETRY & METROLOGY (formerly alignment.py)
# ===========================================================================

def get_aligned_tile(src_img, alpha, scale_factor=1.0):
    h, w = src_img.shape[:2]
    center = (w / 2.0, h / 2.0)
    angle_deg = alpha * 180.0 / np.pi
    M = cv2.getRotationMatrix2D(center, angle_deg, scale_factor)
    return cv2.warpAffine(
        src_img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


def fit_circle_least_squares(points):
    """Fits a circle to 2D coordinates using algebraic least squares."""
    pts = np.array(points, dtype=np.float64)
    A = np.column_stack((2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))))
    B = pts[:, 0] ** 2 + pts[:, 1] ** 2
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    xc, yc = res[0], res[1]

    val = res[2] + xc ** 2 + yc ** 2
    if val < 0:
        raise ValueError("Invalid circle geometry detected during least squares fit.")
    R = np.sqrt(val)
    return xc, yc, R


def robust_circle_fit(points, outlier_pct=0.012, max_iters=5):
    """Fits a circle deterministically, iteratively filtering outliers."""
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("Need at least 3 points to fit a circle.")

    for i in range(max_iters):
        xc, yc, R = fit_circle_least_squares(pts)
        dists = np.linalg.norm(pts - np.array([xc, yc]), axis=1)
        err = np.abs(dists - R)
        threshold = outlier_pct * R
        inliers = pts[err < threshold]

        print(
            f"    [Robust Circle Fit] Iter {i + 1}: Center=({xc:.2f}, {yc:.2f}), R={R:.2f}, "
            f"MeanErr={np.mean(err):.3f}px, MaxErr={np.max(err):.3f}px, "
            f"Inliers={len(inliers)}/{len(pts)}"
        )

        if len(inliers) < 3:
            break
        pts = inliers

    xc, yc, R = fit_circle_least_squares(pts)
    dists = np.linalg.norm(pts - np.array([xc, yc]), axis=1)
    mean_residual = float(np.mean(np.abs(dists - R)))

    # Anisotropy check: compare horizontal vs vertical residual spread
    dx = pts[:, 0] - xc
    dy = pts[:, 1] - yc
    h_mask = np.abs(dx) > np.abs(dy)
    v_mask = ~h_mask
    if h_mask.sum() > 5 and v_mask.sum() > 5:
        h_res = np.mean(np.abs(np.abs(np.sqrt(dx[h_mask] ** 2 + dy[h_mask] ** 2)) - R))
        v_res = np.mean(np.abs(np.abs(np.sqrt(dx[v_mask] ** 2 + dy[v_mask] ** 2)) - R))
        
        if max(h_res, v_res) > 1.5:
            ratio = max(h_res, v_res) / (min(h_res, v_res) + 1e-9)
            if ratio > 1.002:
                print(
                    f"  [Anisotropy Warning] H-residual={h_res:.4f}px, V-residual={v_res:.4f}px, "
                    f"Ratio={ratio:.4f} — possible non-square pixel pitch."
                )

    return xc, yc, R, mean_residual


def find_dynamic_clipping_bounds(contour_pts, ds_factor, base_tolerance_px=3):
    """Identifies where wafer bounds are truncated or clipped."""
    tolerance_px = int(np.clip(base_tolerance_px / ds_factor, 1, 20))

    x_coords = contour_pts[:, 0]
    y_coords = contour_pts[:, 1]
    left_edge = np.min(x_coords)
    right_edge = np.max(x_coords)
    top_edge = np.min(y_coords)
    bottom_edge = np.max(y_coords)

    left_clip_count = np.sum(x_coords < (left_edge + tolerance_px))
    right_clip_count = np.sum(x_coords > (right_edge - tolerance_px))
    top_clip_count = np.sum(y_coords < (top_edge + tolerance_px))
    bottom_clip_count = np.sum(y_coords > (bottom_edge - tolerance_px))

    dynamic_left_limit = left_edge + tolerance_px if left_clip_count > 15 else left_edge - 1
    dynamic_right_limit = right_edge - tolerance_px if right_clip_count > 15 else right_edge + 1
    dynamic_top_limit = top_edge + tolerance_px if top_clip_count > 15 else top_edge - 1
    dynamic_bottom_limit = bottom_edge + 1

    print(f"  [Clipping Diagnostic] Scaled tolerance: {tolerance_px}px (ds_factor={ds_factor})")
    print(
        f"  [Clipping Diagnostic] Dynamic Limits: L={dynamic_left_limit:.1f}, "
        f"R={dynamic_right_limit:.1f}, T={dynamic_top_limit:.1f}, B={dynamic_bottom_limit:.1f}"
    )

    return dynamic_left_limit, dynamic_right_limit, dynamic_top_limit, dynamic_bottom_limit


def robust_vertical_profile_flat(gray_img, xc_ds, yc_ds, R_ds, h_img, w_img, is_inverted=False):
    """Scans vertical columns using gradient-of-intensity thresholding."""
    profile_pts = []
    x_start = int(xc_ds - 0.20 * R_ds)
    x_end = int(xc_ds + 0.20 * R_ds)

    y_start_scan = int(yc_ds + 0.75 * R_ds)
    y_end_scan = int(yc_ds + 0.94 * R_ds)
    y_end_scan = min(y_end_scan, h_img - 15)

    for x in range(x_start, x_end, 2):
        if x < 0 or x >= w_img:
            continue

        column_pixels = gray_img[y_start_scan:y_end_scan, x].astype(np.float64)
        if len(column_pixels) < 10:
            continue

        gradients = np.diff(column_pixels)

        for i in range(len(gradients)):
            intensity = column_pixels[i]
            next_intensity = column_pixels[i + 1]
            grad = gradients[i]

            if not is_inverted:
                if grad < -5.0 and intensity > 90 and (15 <= next_intensity <= 75):
                    y_edge = y_start_scan + i + 1
                    profile_pts.append([x, y_edge])
                    break
            else:
                if grad > 5.0 and intensity < 70 and (90 <= next_intensity <= 160):
                    y_edge = y_start_scan + i + 1
                    profile_pts.append([x, y_edge])
                    break

    profile_pts = np.array(profile_pts, dtype=np.float64)

    if len(profile_pts) >= 20:
        slope, intercept = np.polyfit(profile_pts[:, 0], profile_pts[:, 1], 1)
        angle_deg = abs(np.arctan(slope) * 180.0 / np.pi)

        if angle_deg < 0.5:
            print(f"  [Vertical Profile Info] Discarded edge alignment with slope {angle_deg:.3f}° as canvas padding.")
            return np.array([])

    return profile_pts


def _weighted_line_fit(pts):
    """Fits a line to pts with weights proportional to distance from sector boundary."""
    if len(pts) < 2:
        raise ValueError("Need at least 2 points for line fit.")
    cx = np.mean(pts[:, 0])
    weights = 1.0 / (1.0 + np.abs(pts[:, 0] - cx) / (np.ptp(pts[:, 0]) + 1e-9))
    coeffs = np.polyfit(pts[:, 0], pts[:, 1], 1, w=weights)
    slope, intercept = coeffs
    predicted = slope * pts[:, 0] + intercept
    mean_err = float(np.mean(np.abs(pts[:, 1] - predicted)))
    return slope, intercept, mean_err


def geometry_based_flat_fit(circular_pts, xc_ds, yc_ds, R_ds):
    """Isolates flat points using a tight radial band-pass + angular sector."""
    dx = circular_pts[:, 0] - xc_ds
    dy = circular_pts[:, 1] - yc_ds
    dists = np.sqrt(dx ** 2 + dy ** 2)
    angles = np.arctan2(dy, dx)

    best_alpha = None
    best_residual = np.inf
    best_flat_pts = None

    for sector_half in np.arange(0.44, 0.19, -0.04):
        bottom_sector_mask = (
            (angles >= np.pi / 2.0 - sector_half)
            & (angles <= np.pi / 2.0 + sector_half)
        )
        flat_chord_mask = (
            bottom_sector_mask
            & (dists >= 0.91 * R_ds)
            & (dists <= 0.995 * R_ds)
        )
        flat_pts = circular_pts[flat_chord_mask]

        if len(flat_pts) < 10:
            continue

        for _ in range(5):
            if len(flat_pts) < 10:
                break
            slope, intercept, mean_err = _weighted_line_fit(flat_pts)
            predicted_y = slope * flat_pts[:, 0] + intercept
            errors = np.abs(flat_pts[:, 1] - predicted_y)
            inliers = errors < 0.8
            if np.sum(inliers) < 10:
                break
            flat_pts = flat_pts[inliers]

        if len(flat_pts) < 10:
            continue

        slope, intercept, residual = _weighted_line_fit(flat_pts)
        alpha_candidate = np.arctan(slope)

        if residual < best_residual:
            best_residual = residual
            best_alpha = alpha_candidate
            best_flat_pts = flat_pts.copy()

    return best_flat_pts, best_alpha, best_residual


def detect_wafer_on_canvas(ds_image, ds_factor):
    """Segments and extracts wafer boundary, center, and rotation metrics."""
    h_img, w_img = ds_image.shape[:2]
    gray = cv2.cvtColor(ds_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = float(max(15.0, otsu_thresh))
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    cw = int(w_img * 0.05)
    ch = int(h_img * 0.05)
    corners = [
        thresh[0:ch, 0:cw],
        thresh[0:ch, w_img-cw:w_img],
        thresh[h_img-ch:h_img, 0:cw],
        thresh[h_img-ch:h_img, w_img-cw:w_img]
    ]
    corner_pixels = np.concatenate([c.flatten() for c in corners])
    corner_white_pct = (np.sum(corner_pixels == 255) / corner_pixels.size) * 100.0

    is_inverted = False
    if corner_white_pct > 50.0:
        thresh = cv2.bitwise_not(thresh)
        is_inverted = True

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    if num_labels < 2:
        _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
        corner_pixels_fb = np.concatenate([c.flatten() for c in [
            thresh[0:ch, 0:cw],
            thresh[0:ch, w_img-cw:w_img],
            thresh[h_img-ch:h_img, 0:cw],
            thresh[h_img-ch:h_img, w_img-cw:w_img]
        ]])
        corner_white_pct_fb = (np.sum(corner_pixels_fb == 255) / corner_pixels_fb.size) * 100.0
        if corner_white_pct_fb > 50.0:
            thresh = cv2.bitwise_not(thresh)
            is_inverted = True
        else:
            is_inverted = False
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    if num_labels >= 2:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        wafer_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        wafer_mask = thresh

    k_size = int(np.clip(15 * (ds_factor / 0.05), 5, 31))
    if k_size % 2 == 0:
        k_size += 1
    kernel_seg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    eroded_mask = cv2.erode(wafer_mask, kernel_seg)

    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(wafer_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [largest_contour], -1, 255, -1)
    else:
        solid_mask = wafer_mask

    solid_mask = cv2.dilate(solid_mask, kernel_seg)
    solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, kernel_seg)
    solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_OPEN, kernel_seg)

    contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Could not find wafer boundary contour on stitched canvas.")

    wafer_contour = max(contours, key=cv2.contourArea)
    contour_pts = wafer_contour.squeeze(axis=1).astype(np.float64)

    left_lim, right_lim, top_lim, bottom_lim = find_dynamic_clipping_bounds(contour_pts, ds_factor)
    boundary_mask = (
        (contour_pts[:, 0] > left_lim)
        & (contour_pts[:, 0] < right_lim)
        & (contour_pts[:, 1] > top_lim)
        & (contour_pts[:, 1] < bottom_lim)
    )
    circular_pts = contour_pts[boundary_mask]

    if len(circular_pts) < 10:
        circular_pts = contour_pts

    rough_yc = np.mean(circular_pts[:, 1])
    top_half_pts = circular_pts[circular_pts[:, 1] < rough_yc]

    if len(top_half_pts) < 3:
        top_half_pts = circular_pts

    xc_ds, yc_ds, R_ds, circle_residual = robust_circle_fit(top_half_pts, outlier_pct=0.012, max_iters=5)

    # Core Detection Strategy 1: Vertical scanning profile
    profile_pts = robust_vertical_profile_flat(gray, xc_ds, yc_ds, R_ds, h_img, w_img, is_inverted=is_inverted)
    if len(profile_pts) >= 20:
        for _ in range(5):
            slope, intercept = np.polyfit(profile_pts[:, 0], profile_pts[:, 1], 1)
            predicted_y = slope * profile_pts[:, 0] + intercept
            errors = np.abs(profile_pts[:, 1] - predicted_y)
            inliers = errors < 1.2
            if np.sum(inliers) < 15:
                break
            profile_pts = profile_pts[inliers]

        slope_profile, intercept_profile = np.polyfit(profile_pts[:, 0], profile_pts[:, 1], 1)
        alpha = np.arctan(slope_profile)
        return xc_ds / ds_factor, yc_ds / ds_factor, R_ds / ds_factor, alpha

    # Core Detection Strategy 2: Radial geometry bounds
    flat_pts, alpha, fit_residual = geometry_based_flat_fit(circular_pts, xc_ds, yc_ds, R_ds)
    if flat_pts is not None and len(flat_pts) >= 10:
        return xc_ds / ds_factor, yc_ds / ds_factor, R_ds / ds_factor, alpha

    # Core Detection Strategy 3: Polar grid sector sweep
    dx = circular_pts[:, 0] - xc_ds
    dy = circular_pts[:, 1] - yc_ds
    dists = np.sqrt(dx ** 2 + dy ** 2)
    angles_all = np.arctan2(dy, dx)

    num_steps = 360
    search_angles = np.linspace(np.pi / 2.0 - 0.44, np.pi / 2.0 + 0.44, num_steps)
    window_half_width = 0.22
    avg_dists = np.zeros(num_steps)
    for idx, angle in enumerate(search_angles):
        angle_diff = np.abs(angles_all - angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        in_window_mask = angle_diff <= window_half_width
        avg_dists[idx] = np.mean(dists[in_window_mask]) if np.any(in_window_mask) else R_ds

    min_idx = np.argmin(avg_dists)
    flat_angle_rad = search_angles[min_idx]
    alpha = flat_angle_rad - np.pi / 2.0
    return xc_ds / ds_factor, yc_ds / ds_factor, R_ds / ds_factor, alpha


# ===========================================================================
# 3. GDS PARSING HELPER MODULES (formerly gds_parser.py)
# ===========================================================================

def parse_gds_wafer_boundary(path, layer=0, datatype=0):
    """Parses design file geometry boundaries to find absolute radius."""
    lib = gdstk.read_gds(path)
    top_cells = lib.top_level()
    if not top_cells:
        raise ValueError("No top-level cells found in GDS.")
    cell = top_cells[0]

    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapezoid is None:
        raise ImportError("Neither np.trapezoid nor np.trapz was found in this NumPy installation.")

    best = None
    best_area = 0
    for poly in cell.get_polygons(layer=layer, datatype=datatype):
        pts = poly.points
        area = float(np.abs(trapezoid(pts[:, 1], pts[:, 0])))
        if area > best_area:
            best_area = area
            best = pts

    if best is None:
        raise ValueError(f"No polygons found on layer {layer} datatype {datatype}.")

    xc = float(np.mean(best[:, 0]))
    yc = float(np.mean(best[:, 1]))
    R  = float(np.mean(np.sqrt((best[:, 0] - xc)**2 + (best[:, 1] - yc)**2)))
    return xc, yc, R


def get_layer0_polygons(path):
    """Retrieves all polygon geometries sitting on GDS layer 0."""
    lib = gdstk.read_gds(path)
    top_cells = lib.top_level()
    if not top_cells:
        raise ValueError("No top-level cells found in GDS.")
    cell = top_cells[0]

    polygons = []
    for poly in cell.get_polygons():
        if poly.layer == 0:
            polygons.append(poly.points.astype(np.float64))

    return polygons


# ===========================================================================
# 4. DOWNSCALED STITCHING OPERATIONS (formerly stitcher.py)
# ===========================================================================

def generate_downscaled_stitch(folder, config):
    """Reassembles global unaligned preview using downscaled tile steps."""
    folder_path = Path(folder)
    tile_files = list(folder_path.glob("tile_x*_y*.*"))
    if not tile_files:
        raise ValueError(f"No grid tile files found in: {folder}")

    tile_ext = tile_files[0].suffix
    cols = config["tile_cols"]
    rows = config["tile_rows"]
    tw = config["tile_width"]
    th = config["tile_height"]
    ds = config["downscale_factor"]

    overlap_x = config["overlap_x_percent"] / 100.0
    overlap_y = config["overlap_y_percent"] / 100.0
    step_x = tw * (1.0 - overlap_x)
    step_y = th * (1.0 - overlap_y)

    canvas_w = int(((cols - 1) * step_x + tw) * ds)
    canvas_h = int(((rows - 1) * step_y + th) * ds)

    ds_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    total_tiles = len(tile_files)
    last_printed_pct = 0

    print("[Downscaled Stitch] Progress: 0%")

    for idx, tile_file in enumerate(tile_files):
        pct = int(((idx + 1) / total_tiles) * 100)
        decile = pct // 10
        if decile > last_printed_pct and decile <= 10:
            print(f"[Downscaled Stitch] Progress: {decile * 10}%")
            last_printed_pct = decile

        name = tile_file.stem
        match = re.search(r'tile_x(\d+)_y(\d+)', name)
        if not match:
            continue
        col, row = int(match.group(1)), int(match.group(2))

        img = cv2.imread(str(tile_file))
        if img is None:
            continue

        img_ds = cv2.resize(img, (int(tw * ds), int(th * ds)), interpolation=cv2.INTER_AREA)
        x_can = max(0, int(((col - 1) * step_x) * ds))
        y_can = max(0, int(((row - 1) * step_y) * ds))

        h_ds, w_ds = img_ds.shape[:2]
        h_ds_clamp = min(h_ds, canvas_h - y_can)
        w_ds_clamp = min(w_ds, canvas_w - x_can)

        if h_ds_clamp > 0 and w_ds_clamp > 0:
            ds_canvas[y_can:y_can + h_ds_clamp, x_can:x_can + w_ds_clamp] = (
                img_ds[:h_ds_clamp, :w_ds_clamp]
            )

    return ds_canvas, tile_ext


# ===========================================================================
# 5. COORDINATE TRANSFORMATION PIPELINE (formerly transformer.py)
# ===========================================================================

class WaferTransformer:
    """Handles isomorphic bidirection coordinate mappings: canvas px <=> GDS um."""
    def __init__(
        self,
        canvas_center: tuple[float, float],
        canvas_radius: float,
        canvas_flat_angle: float,
        gds_radius: float,
        config: dict,
        ext: str,
        exclusions: set | None = None,
        S_x: float | None = None,
        S_y: float | None = None,
        shear: float = 0.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        map_mode: bool = False,
        gds_center: tuple[float, float] = (0.0, 0.0),
    ):
        self.xc = float(canvas_center[0])
        self.yc = float(canvas_center[1])
        self.Rc = float(canvas_radius)
        self.Rg = float(gds_radius)
        self.xg_c = float(gds_center[0])
        self.yg_c = float(gds_center[1])
        self.map_mode = map_mode
        self.alpha = float(canvas_flat_angle)

        S_iso = self.Rg / self.Rc if self.Rc > 0 else 1.0
        self.S_x = float(S_x) if S_x is not None else S_iso
        self.S_y = float(S_y) if S_y is not None else S_iso
        self.S   = (self.S_x + self.S_y) / 2.0

        self.shear = float(shear)
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)

        self.config = config
        self.ext = ext
        self.exclusions = exclusions or set()

        self._tw     = config["tile_width"]
        self._th     = config["tile_height"]
        self._cols   = config["tile_cols"]
        self._rows   = config["tile_rows"]
        self._step_x = self._tw * (1.0 - config["overlap_x_percent"] / 100.0)
        self._step_y = self._th * (1.0 - config["overlap_y_percent"] / 100.0)
        self._out_size = config.get("output_image_size", 4000)

        self._cos_a = np.cos(self.alpha)
        self._sin_a = np.sin(self.alpha)
        self.run_self_test()

    def tile_to_canvas(self, col: int, row: int, local_x: float, local_y: float):
        x_can = (col - 1) * self._step_x + local_x
        y_can = (row - 1) * self._step_y + local_y
        return x_can, y_can

    def canvas_to_gds(self, x_can: float, y_can: float, debug: bool = False):
        dx = x_can - self.xc
        dy = self.yc - y_can

        dx_rot = dx * self._cos_a - dy * self._sin_a
        dy_rot = dx * self._sin_a + dy * self._cos_a

        gy_eff = dy_rot * self.S_y + self.y_offset
        gx_eff = dx_rot * self.S_x + self.shear * (dy_rot * self.S_y) + self.x_offset

        gx_centered = -gx_eff
        gy_centered = gy_eff

        x_gds = gx_centered + self.xg_c
        y_gds = gy_centered + self.yg_c
        return x_gds, y_gds

    def gds_to_canvas(self, x_gds: float, y_gds: float, debug: bool = False):
        gx_centered = x_gds - self.xg_c
        gy_centered = y_gds - self.yg_c

        gx_eff = -gx_centered
        gy_eff = gy_centered

        dy_rot = (gy_eff - self.y_offset) / self.S_y
        dx_rot = ((gx_eff - self.x_offset) - self.shear * (gy_eff - self.y_offset)) / self.S_x

        dx = dx_rot * self._cos_a + dy_rot * self._sin_a
        dy = -dx_rot * self._sin_a + dy_rot * self._cos_a

        x_can = self.xc + dx
        y_can = self.yc - dy
        return x_can, y_can

    def transform_gds_to_target_img(self, x_gds: float, y_gds: float, out_size: int):
        half  = out_size / 2.0
        scale = (0.925 * half) / self.Rg
        gx_centered = -(x_gds - self.xg_c)
        x_img = half + gx_centered * scale
        y_img = half - (y_gds - self.yg_c) * scale
        return x_img, y_img

    def is_tile_fully_inside(self, col: int, row: int, radius_fraction: float = 0.98) -> bool:
        tile_key = f"tile_x{col:03d}_y{row:03d}{self.ext}"
        if tile_key in self.exclusions:
            return False
        cx_can, cy_can = self.tile_to_canvas(col, row, self._tw / 2.0, self._th / 2.0)
        x_gds, y_gds  = self.canvas_to_gds(cx_can, cy_can)
        return np.sqrt((x_gds - self.xg_c) ** 2 + (y_gds - self.yg_c) ** 2) <= radius_fraction * self.Rg

    def run_self_test(self):
        """Verifies mathematical conversion reversibility."""
        test_gds_points = [(0.0, 0.0), (1500.0, -1500.0), (-self.Rg * 0.5, self.Rg * 0.5)]
        passed_all = True
        for gx, gy in test_gds_points:
            cx, cy = self.gds_to_canvas(gx, gy)
            gx_rt, gy_rt = self.canvas_to_gds(cx, cy)
            if abs(gx - gx_rt) > 1e-4 or abs(gy - gy_rt) > 1e-4:
                passed_all = False

        test_can_points = [(self.xc, self.yc), (self.xc + self.Rc * 0.4, self.yc - self.Rc * 0.4)]
        for cx, cy in test_can_points:
            gx, gy = self.canvas_to_gds(cx, cy)
            cx_rt, cy_rt = self.gds_to_canvas(gx, gy)
            if abs(cx - cx_rt) > 1e-3 or abs(cy - cy_rt) > 1e-3:
                passed_all = False
                
        if passed_all:
            print("[Transformer Self-Test] Success. Dynamic mappings are completely isomorphic.")


# ===========================================================================
# 6. MANUAL OVERLAY ALIGNMENT TOOL (formerly manual_align_gui.py)
# ===========================================================================

class ManualAlignApp:
    def __init__(self, root, ds_image, xc_ds, yc_ds, R_ds, ds_factor, gds_polygons, gds_R, initial_angle_rad, map_mode=False, gds_center=(0.0, 0.0), shear=0.0):
        self.root = root
        self.ds_image = ds_image
        self.xc_ds = xc_ds
        self.yc_ds = yc_ds
        self.R_ds = R_ds
        self.ds_factor = ds_factor
        self.gds_polygons = gds_polygons
        self.Rg = gds_R
        self.map_mode = map_mode
        self.xg_c = float(gds_center[0])
        self.yg_c = float(gds_center[1])
        self.shear = float(shear)

        self.initial_angle_rad = initial_angle_rad
        self.initial_angle_deg = initial_angle_rad * 180.0 / np.pi
        self.current_angle_deg = self.initial_angle_deg
        self.current_angle_rad = self.initial_angle_rad

        self.offset_x = 0.0
        self.offset_y = 0.0
        self.scale_mult = 1.0

        self.V_w = 1000
        self.V_h = 600

        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.show_gds = True
        self.setup_ui()
        self.setup_bindings()
        self.render()

    def setup_ui(self):
        self.root.title("Interactive GDS Alignment Calibration Workspace")
        self.root.geometry("1400x700")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")

        self.sidebar = ttk.Frame(self.root, padding=15, width=360)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        self.sidebar.grid_propagate(False)

        self.canvas = tk.Canvas(self.root, width=self.V_w, height=self.V_h, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=10, pady=10)

        # UI Controls
        ttk.Label(self.sidebar, text="1. Rotation (Degrees)", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 2))
        rot_frame = ttk.Frame(self.sidebar)
        rot_frame.pack(fill="x", pady=(0, 10))
        self.rot_var = tk.StringVar(value=f"{self.current_angle_deg:.3f}")
        self.rot_entry = ttk.Entry(rot_frame, textvariable=self.rot_var, width=12)
        self.rot_entry.pack(side="left", padx=(0, 5))
        ttk.Button(rot_frame, text="Apply", command=self.apply_rotation).pack(side="left")

        rot_btn_frame = ttk.Frame(self.sidebar)
        rot_btn_frame.pack(fill="x", pady=(0, 15))
        ttk.Button(rot_btn_frame, text="-0.10°", width=7, command=lambda: self.adjust_angle(-0.1)).grid(row=0, column=0, padx=2)
        ttk.Button(rot_btn_frame, text="-0.01°", width=7, command=lambda: self.adjust_angle(-0.01)).grid(row=0, column=1, padx=2)
        ttk.Button(rot_btn_frame, text="+0.01°", width=7, command=lambda: self.adjust_angle(0.01)).grid(row=0, column=2, padx=2)
        ttk.Button(rot_btn_frame, text="+0.10°", width=7, command=lambda: self.adjust_angle(0.1)).grid(row=0, column=3, padx=2)

        ttk.Label(self.sidebar, text="2. Translation X (microns)", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 2))
        tx_frame = ttk.Frame(self.sidebar)
        tx_frame.pack(fill="x", pady=(0, 10))
        self.tx_var = tk.StringVar(value=f"{self.offset_x:.1f}")
        self.tx_entry = ttk.Entry(tx_frame, textvariable=self.tx_var, width=12)
        self.tx_entry.pack(side="left", padx=(0, 5))
        ttk.Button(tx_frame, text="Apply", command=self.apply_tx).pack(side="left")

        tx_btn_frame = ttk.Frame(self.sidebar)
        tx_btn_frame.pack(fill="x", pady=(0, 15))
        ttk.Button(tx_btn_frame, text="-50µm", width=7, command=lambda: self.adjust_tx(-50.0)).grid(row=0, column=0, padx=2)
        ttk.Button(tx_btn_frame, text="-5µm", width=7, command=lambda: self.adjust_tx(-5.0)).grid(row=0, column=1, padx=2)
        ttk.Button(tx_btn_frame, text="+5µm", width=7, command=lambda: self.adjust_tx(5.0)).grid(row=0, column=2, padx=2)
        ttk.Button(tx_btn_frame, text="+50µm", width=7, command=lambda: self.adjust_tx(50.0)).grid(row=0, column=3, padx=2)

        ttk.Label(self.sidebar, text="3. Translation Y (microns)", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 2))
        ty_frame = ttk.Frame(self.sidebar)
        ty_frame.pack(fill="x", pady=(0, 10))
        self.ty_var = tk.StringVar(value=f"{self.offset_y:.1f}")
        self.ty_entry = ttk.Entry(ty_frame, textvariable=self.ty_var, width=12)
        self.ty_entry.pack(side="left", padx=(0, 5))
        ttk.Button(ty_frame, text="Apply", command=self.apply_ty).pack(side="left")

        ty_btn_frame = ttk.Frame(self.sidebar)
        ty_btn_frame.pack(fill="x", pady=(0, 15))
        ttk.Button(ty_btn_frame, text="-50µm", width=7, command=lambda: self.adjust_ty(-50.0)).grid(row=0, column=0, padx=2)
        ttk.Button(ty_btn_frame, text="-5µm", width=7, command=lambda: self.adjust_ty(-5.0)).grid(row=0, column=1, padx=2)
        ttk.Button(ty_btn_frame, text="+5µm", width=7, command=lambda: self.adjust_ty(5.0)).grid(row=0, column=2, padx=2)
        ttk.Button(ty_btn_frame, text="+50µm", width=7, command=lambda: self.adjust_ty(50.0)).grid(row=0, column=3, padx=2)

        ttk.Label(self.sidebar, text="4. Scale Factor Multiplier", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 2))
        sc_frame = ttk.Frame(self.sidebar)
        sc_frame.pack(fill="x", pady=(0, 10))
        self.scale_var = tk.StringVar(value=f"{self.scale_mult:.5f}")
        self.scale_entry = ttk.Entry(sc_frame, textvariable=self.scale_var, width=12)
        self.scale_entry.pack(side="left", padx=(0, 5))
        ttk.Button(sc_frame, text="Apply", command=self.apply_scale).pack(side="left")

        sc_btn_frame = ttk.Frame(self.sidebar)
        sc_btn_frame.pack(fill="x", pady=(0, 20))
        ttk.Button(sc_btn_frame, text="-0.50%", width=7, command=lambda: self.adjust_scale(-0.005)).grid(row=0, column=0, padx=2)
        ttk.Button(sc_btn_frame, text="-0.05%", width=7, command=lambda: self.adjust_scale(-0.0005)).grid(row=0, column=1, padx=2)
        ttk.Button(sc_btn_frame, text="+0.05%", width=7, command=lambda: self.adjust_scale(0.0005)).grid(row=0, column=2, padx=2)
        ttk.Button(sc_btn_frame, text="+0.50%", width=7, command=lambda: self.adjust_scale(0.005)).grid(row=0, column=3, padx=2)

        self.status_label = ttk.Label(self.sidebar, text="", font=("Helvetica", 9, "bold"), justify="left")
        self.status_label.pack(anchor="w", pady=(0, 25))

        ttk.Button(self.sidebar, text="✔ Confirm Settings", command=self.confirm).pack(fill="x", pady=4)
        ttk.Button(self.sidebar, text="✖ Cancel (No Calibration)", command=self.cancel).pack(fill="x", pady=4)

    def setup_bindings(self):
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, 'is_dragging', False))
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Button-4>", lambda e: self.adjust_zoom(1.15))
        self.canvas.bind("<Button-5>", lambda e: self.adjust_zoom(1.0 / 1.15))

        self.rot_entry.bind("<Return>", lambda e: self.apply_rotation())
        self.tx_entry.bind("<Return>", lambda e: self.apply_tx())
        self.ty_entry.bind("<Return>", lambda e: self.apply_ty())
        self.scale_entry.bind("<Return>", lambda e: self.apply_scale())

        self.root.bind("<Control-equal>", lambda e: self.adjust_zoom(1.15))
        self.root.bind("<Control-plus>", lambda e: self.adjust_zoom(1.15))
        self.root.bind("<Control-minus>", lambda e: self.adjust_zoom(1.0 / 1.15))
        self.root.bind("<KeyPress-r>", lambda e: self.reset_viewport())

    def apply_rotation(self):
        try:
            val = float(self.rot_var.get())
            if -45.0 <= val <= 45.0:
                self.current_angle_deg = val
                self.current_angle_rad = val * np.pi / 180.0
                self.render()
        except ValueError:
            pass

    def adjust_angle(self, offset):
        self.current_angle_deg = np.clip(self.current_angle_deg + offset, -45.0, 45.0)
        self.current_angle_rad = self.current_angle_deg * np.pi / 180.0
        self.rot_var.set(f"{self.current_angle_deg:.3f}")
        self.render()

    def apply_tx(self):
        try:
            self.offset_x = float(self.tx_var.get())
            self.render()
        except ValueError:
            pass

    def adjust_tx(self, offset):
        self.offset_x += offset
        self.tx_var.set(f"{self.offset_x:.1f}")
        self.render()

    def apply_ty(self):
        try:
            self.offset_y = float(self.ty_var.get())
            self.render()
        except ValueError:
            pass

    def adjust_ty(self, offset):
        self.offset_y += offset
        self.ty_var.set(f"{self.offset_y:.1f}")
        self.render()

    def apply_scale(self):
        try:
            val = float(self.scale_var.get())
            if 0.5 <= val <= 2.0:
                self.scale_mult = val
                self.render()
        except ValueError:
            pass

    def adjust_scale(self, offset):
        self.scale_mult = np.clip(self.scale_mult + offset, 0.5, 2.0)
        self.scale_var.set(f"{self.scale_mult:.5f}")
        self.render()

    def on_drag_start(self, event):
        self.is_dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.orig_pan_x = self.pan_x
        self.orig_pan_y = self.pan_y

    def on_drag(self, event):
        if self.is_dragging:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y

            H, W = self.ds_image.shape[:2]
            S_base = min(self.V_w / W, self.V_h / H)
            S = S_base * self.zoom

            self.pan_x = self.orig_pan_x - dx / S
            self.pan_y = self.orig_pan_y - dy / S
            self.render()

    def on_scroll(self, event):
        if event.delta > 0:
            self.adjust_zoom(1.15)
        else:
            self.adjust_zoom(1.0 / 1.15)

    def adjust_zoom(self, factor):
        self.zoom = np.clip(self.zoom * factor, 1.0, 15.0)
        self.render()

    def reset_viewport(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render()

    def render(self):
        H, W = self.ds_image.shape[:2]
        S_base = min(self.V_w / W, self.V_h / H)
        S = S_base * self.zoom

        cx_m, cy_m = self.xc_ds, self.yc_ds
        M_rot = cv2.getRotationMatrix2D((cx_m, cy_m), self.current_angle_deg, 1.0)
        T_rot = np.eye(3)
        T_rot[:2, :] = M_rot

        K = np.array([cx_m, cy_m, 1.0])
        K_rot = T_rot @ K
        K_rx, K_ry = K_rot[0], K_rot[1]

        tx = self.V_w / 2.0 - S * (K_rx + self.pan_x)
        ty = self.V_h / 2.0 - S * (K_ry + self.pan_y)

        T_view = np.array([[S, 0, tx], [0, S, ty], [0, 0, 1]])
        T_final = T_view @ T_rot
        M_final = T_final[:2, :]

        display_img = cv2.warpAffine(
            self.ds_image, M_final, (self.V_w, self.V_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30)
        )

        if self.show_gds and self.gds_polygons:
            S_x = (self.Rg / self.R_ds) * self.scale_mult
            S_y = (self.Rg / self.R_ds) * self.scale_mult

            for poly in self.gds_polygons:
                if len(poly) > 300:
                    poly = poly[::max(1, len(poly) // 300)]

                pts_scr = []
                for gx, gy in poly:
                    gx_eff = -(gx - self.xg_c)
                    gx_off = gx_eff - self.offset_x
                    gy_off = (gy - self.yg_c) - self.offset_y

                    dy = gy_off / S_y
                    dx = (gx_off - self.shear * gy_off) / S_x

                    x_can = dx + self.xc_ds
                    y_can = self.yc_ds - dy

                    scr_x = int(S * x_can + tx)
                    scr_y = int(S * y_can + ty)

                    if -500 <= scr_x < self.V_w + 500 and -500 <= scr_y < self.V_h + 500:
                        pts_scr.append([scr_x, scr_y])

                if len(pts_scr) >= 2:
                    pts_scr = np.array(pts_scr, dtype=np.int32)
                    cv2.polylines(display_img, [pts_scr], isClosed=True, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        for y_line in [180, 300, 420]:
            cv2.line(display_img, (0, y_line), (self.V_w, y_line), (0, 255, 255), 1, cv2.LINE_AA)

        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        status_text = (
            f"Angle: {self.current_angle_deg:.3f}°\n"
            f"Zoom: {self.zoom:.2f}x\n"
            f"Pan: ({int(self.pan_x)}, {int(self.pan_y)})\n"
            f"Mode: Alignment Active (Mirroring)"
        )
        self.status_label.config(text=status_text)

    def confirm(self):
        self.result_angle_rad = self.current_angle_rad
        self.result_tx = self.offset_x
        self.result_ty = self.offset_y
        self.result_scale_mult = self.scale_mult
        self.root.destroy()

    def cancel(self):
        self.result_angle_rad = self.initial_angle_rad
        self.result_tx = 0.0
        self.result_ty = 0.0
        self.result_scale_mult = 1.0
        self.root.destroy()


def run_manual_alignment(ds_canvas, config, xc_ds, yc_ds, R_ds, ds_factor, tile_ext, initial_angle_rad, gds_polygons, gds_R, map_mode=False, gds_center=(0.0, 0.0), shear=0.0):
    root = tk.Tk()
    app = ManualAlignApp(
        root, ds_canvas, xc_ds, yc_ds, R_ds, ds_factor,
        gds_polygons, gds_R, initial_angle_rad, map_mode=map_mode,
        gds_center=gds_center, shear=shear
    )
    root.protocol("WM_DELETE_WINDOW", app.cancel)
    root.mainloop()
    return app.result_angle_rad, app.result_tx, app.result_ty, app.result_scale_mult


# ===========================================================================
# 7. INTERACTIVE DEVICE DEFECT MAPPER TOOL (formerly extract_cells.py block)
# ===========================================================================

# Bounding box limits for monitors
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 750

CLASS_COLORS = {
    "blister": (0, 255, 0),       # Green
    "tear": (255, 0, 0),          # Blue (BGR)
    "delamination": (255, 0, 255),# Magenta
    "particulate": (0, 0, 255),   # Red
    "hole": (0, 255, 255)         # Yellow
}

KEY_MAPPING = {
    ord('1'): "blister",
    ord('2'): "tear",
    ord('3'): "delamination",
    ord('4'): "particulate",
    ord('5'): "hole"
}

LEFT_ARROW_CODES = [2424832, 65361, 81, 0x250000, 63234]  
RIGHT_ARROW_CODES = [2555904, 65363, 83, 0x270000, 63235]


class DeviceDefectMapperTool:
    def __init__(self, wafer_id, cells, out_dir, transformer, gds_R, config):
        self.wafer_id = wafer_id
        self.cells = cells
        self.out_dir = Path(out_dir)
        self.transformer = transformer
        self.gds_R = gds_R
        self.config = config
        
        # Self-scaled display window width leaves horizontal room for 320px sidebar
        self.max_disp_w = 950
        self.max_disp_h = MAX_DISPLAY_HEIGHT
        
        self.output_json_path = f"{wafer_id}_device_defects.json"
        self.output_stitch_path = f"{wafer_id}_stitched_devices.jpg"

        self.cell_files = []
        for cell in self.cells:
            filename = f"{wafer_id}_cell_{cell['row']}-{cell['col']}.jpg"
            filepath = self.out_dir / filename
            if filepath.exists():
                self.cell_files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "cell_data": cell
                })

        if not self.cell_files:
            raise FileNotFoundError(f"No cell crops discovered in '{self.out_dir}'. Ensure step 1 extraction succeeded first.")

        self.cell_files.sort(key=lambda x: (x["cell_data"]["row"], x["cell_data"]["col"]))

        # Cache absolute GDS bounds across all devices to normalize wafer map coords
        self.all_min_x = min(c["cell_data"]["bbox"][0] for c in self.cell_files)
        self.all_min_y = min(c["cell_data"]["bbox"][1] for c in self.cell_files)
        self.all_max_x = max(c["cell_data"]["bbox"][2] for c in self.cell_files)
        self.all_max_y = max(c["cell_data"]["bbox"][3] for c in self.cell_files)
        self.gds_w = self.all_max_x - self.all_min_x
        self.gds_h = self.all_max_y - self.all_min_y
        self.cell_map_bounds = []

        self.native_dims = {}
        for entry in self.cell_files:
            fname = entry["filename"]
            fpath = entry["filepath"]
            try:
                with Image.open(fpath) as img_hdr:
                    self.native_dims[fname] = img_hdr.size
            except Exception:
                self.native_dims[fname] = (self.max_disp_w, self.max_disp_h)

        self.current_idx = 0
        self.annotations = self.load_existing_annotations()
        self.exclusions = self.load_exclusions_file()

        self.img_orig = None
        self.img_disp = None
        self.canvas = None
        self.orig_h = 0
        self.orig_w = 0
        self.native_h = 0
        self.native_w = 0
        self.display_width = 0
        self.display_height = 0
        self.scale = 1.0

        self.drawing = False
        self.start_pt = (0, 0)
        self.current_pt = (0, 0)
        self.is_waiting_for_key = False

        self.load_cell_at_index(self.current_idx)

    def load_existing_annotations(self) -> dict:
        if os.path.exists(self.output_json_path):
            try:
                with open(self.output_json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load previous annotations: {e}")
        return {}

    def save_annotations_to_file(self):
        try:
            with open(self.output_json_path, 'w') as f:
                json.dump(self.annotations, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save annotations JSON: {e}")

    def load_exclusions_file(self) -> set:
        path = Path("manual_exclusions.json")
        if path.exists():
            try:
                with open(path, "r") as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()

    def save_exclusions_file(self):
        try:
            with open("manual_exclusions.json", "w") as f:
                json.dump(sorted(list(self.exclusions)), f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save exclusions: {e}")

    def load_cell_at_index(self, idx: int):
        self.current_idx = idx
        cell_entry = self.cell_files[idx]
        filepath = cell_entry["filepath"]
        cell_data = cell_entry["cell_data"]
        filename = cell_entry["filename"]

        self.native_w, self.native_h = self.native_dims.get(filename, (self.max_disp_w, self.max_disp_h))

        preview_filename = f"{self.wafer_id}_cell_{cell_data['row']}-{cell_data['col']}_preview.jpg"
        preview_path = self.out_dir / "previews" / preview_filename

        if preview_path.exists():
            self.img_orig = cv2.imread(str(preview_path))
        else:
            self.img_orig = cv2.imread(str(filepath))

        if self.img_orig is None:
            print(f"Error loading cell image: {filepath}.")
            return

        self.orig_h, self.orig_w = self.img_orig.shape[:2]

        scale_w = self.max_disp_w / float(self.orig_w)
        scale_h = self.max_disp_h / float(self.orig_h)
        self.scale = min(scale_w, scale_h)

        self.display_width = int(self.orig_w * self.scale)
        self.display_height = int(self.orig_h * self.scale)
        self.img_disp = cv2.resize(self.img_orig, (self.display_width, self.display_height))

        if filename not in self.annotations:
            self.annotations[filename] = []

        self.save_annotations_to_file()
        self.redraw_canvas()

    def redraw_canvas(self):
        panel_width = 320
        self.canvas = np.zeros((self.display_height, self.display_width + panel_width, 3), dtype=np.uint8)
        self.canvas[:, :self.display_width] = self.img_disp

        # Fill background of control panel
        self.canvas[:, self.display_width:] = 40

        cell_entry = self.cell_files[self.current_idx]
        filename = cell_entry["filename"]

        # Draw Title & Navigation State inside side panel
        title_y = 35
        cv2.putText(self.canvas, "DEVICE METRIC STATUS", (self.display_width + 15, title_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"Index: {self.current_idx + 1} / {len(self.cell_files)}", (self.display_width + 15, title_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"File: {filename}", (self.display_width + 15, title_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw Legend block inside side panel
        legend_y = 120
        cv2.putText(self.canvas, "LEGEND GUIDE:", (self.display_width + 15, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        legend_items = [
            ("Current Selected", (0, 165, 255)),
            ("Annotated / Active", (0, 120, 0)),
            ("Excluded / Damaged", (0, 0, 150)),
            ("Unvisited / Pending", (80, 80, 80))
        ]
        for idx_l, (label_txt, l_color) in enumerate(legend_items):
            ly = legend_y + 20 + idx_l * 20
            cv2.rectangle(self.canvas, (self.display_width + 15, ly - 10), (self.display_width + 27, ly + 2), l_color, -1)
            cv2.rectangle(self.canvas, (self.display_width + 15, ly - 10), (self.display_width + 27, ly + 2), (255, 255, 255), 1)
            cv2.putText(self.canvas, label_txt, (self.display_width + 37, ly), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        # Keyboard shortcuts mapping list
        shortcuts_y = legend_y + 115
        cv2.putText(self.canvas, "SHORTCUTS:", (self.display_width + 15, shortcuts_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        shortcuts = [
            "Press [1-5] to assign class",
            "[X]: Toggle Exclusion",
            "[C]: Clear current anomalies",
            "[Right/Space]: Next Cell",
            "[Left]: Previous Cell",
            "[Esc/Q]: Save & Compile Stitch"
        ]
        for idx_s, shortcut_text in enumerate(shortcuts):
            sy = shortcuts_y + 18 + idx_s * 15
            cv2.putText(self.canvas, shortcut_text, (self.display_width + 15, sy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # --- WAFER MAP RENDER BLOCK ---
        map_size = 260
        map_padding = 15
        map_draw_size = map_size - 2 * map_padding
        
        map_x_start = self.display_width + 30
        map_y_start = self.display_height - map_size - 15
        if map_y_start < 250:
            map_y_start = max(250, self.display_height - map_size - 5)

        # Container outline for the wafer map
        cv2.rectangle(self.canvas, (map_x_start - 10, map_y_start - 10), 
                      (map_x_start + map_size - 10, map_y_start + map_size - 10), (30, 30, 30), -1)
        cv2.rectangle(self.canvas, (map_x_start - 10, map_y_start - 10), 
                      (map_x_start + map_size - 10, map_y_start + map_size - 10), (100, 100, 100), 1)
        
        cv2.putText(self.canvas, "WAFER MAP (CLICK TO SWITCH)", (map_x_start - 10, map_y_start - 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        self.cell_map_bounds = []
        for idx, entry in enumerate(self.cell_files):
            cell_data = entry["cell_data"]
            cell_filename = entry["filename"]
            min_x, min_y, max_x, max_y = cell_data["bbox"]
            
            # Map coordinates normalized [0, 1] relative to design bounds
            norm_x1 = (min_x - self.all_min_x) / (self.gds_w + 1e-9)
            norm_y1 = (min_y - self.all_min_y) / (self.gds_h + 1e-9)
            norm_x2 = (max_x - self.all_min_x) / (self.gds_w + 1e-9)
            norm_y2 = (max_y - self.all_min_y) / (self.gds_h + 1e-9)
            
            local_x1 = int(map_x_start + norm_x1 * map_draw_size)
            local_y2 = int(map_y_start + (1.0 - norm_y1) * map_draw_size)
            local_x2 = int(map_x_start + norm_x2 * map_draw_size)
            local_y1 = int(map_y_start + (1.0 - norm_y2) * map_draw_size)
            
            mx1, mx2 = min(local_x1, local_x2), max(local_x1, local_x2)
            my1, my2 = min(local_y1, local_y2), max(local_y1, local_y2)
            
            self.cell_map_bounds.append((mx1, my1, mx2, my2))
            
            is_current = (idx == self.current_idx)
            is_excluded = (cell_filename in self.exclusions)
            has_ann = (len(self.annotations.get(cell_filename, [])) > 0)
            
            if is_current:
                color = (0, 165, 255)
                thickness = -1
            elif is_excluded:
                color = (0, 0, 150)
                thickness = -1
            elif has_ann:
                color = (0, 120, 0)
                thickness = -1
            else:
                color = (80, 80, 80)
                thickness = 1
                
            cv2.rectangle(self.canvas, (mx1, my1), (mx2, my2), color, thickness)
            if thickness == -1:
                cv2.rectangle(self.canvas, (mx1, my1), (mx2, my2), (200, 200, 200) if is_current else (40, 40, 40), 1)

        # Draw exclusion status text at standard bottom corner inside screen space
        if filename in self.exclusions:
            cv2.rectangle(self.canvas, (0, 0), (self.display_width, self.display_height), (0, 0, 255), 4)
            cv2.putText(self.canvas, "MARKED FOR EXCLUSION", (15, self.display_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        image_boxes = self.annotations.get(filename, [])
        for box in image_boxes:
            box_type = box["type"]
            x_tl, y_tl, w, h = box["box_px"]
            
            scale_down_x = self.display_width / float(self.native_w)
            scale_down_y = self.display_height / float(self.native_h)
            
            sc_x1, sc_y1 = int(x_tl * scale_down_x), int(y_tl * scale_down_y)
            sc_x2, sc_y2 = int((x_tl + w) * scale_down_x), int((y_tl + h) * scale_down_y)
            
            color = CLASS_COLORS.get(box_type, (255, 255, 255))
            cv2.rectangle(self.canvas, (sc_x1, sc_y1), (sc_x2, sc_y2), color, 2)
            cv2.putText(self.canvas, f"{box_type} (GDS X:{box.get('center_x_um', 0.0):.1f}, Y:{box.get('center_y_um', 0.0):.1f})", 
                        (sc_x1, sc_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        cv2.imshow("Device Defect Register", self.canvas)

    def handle_mouse(self, event, x, y, flags, param):
        if self.is_waiting_for_key: return

        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= self.display_width:
                # User clicked inside wafer map sidebar panel
                for idx, bounds in enumerate(self.cell_map_bounds):
                    bx1, by1, bx2, by2 = bounds
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        self.save_annotations_to_file()
                        self.load_cell_at_index(idx)
                        return
                return
            else:
                self.drawing = True
                self.start_pt, self.current_pt = (x, y), (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Clamp coordinates inside the bounds of the active cell crop space
                clamped_x = max(0, min(x, self.display_width - 1))
                clamped_y = max(0, min(y, self.display_height - 1))
                self.current_pt = (clamped_x, clamped_y)
                
                temp_frame = self.canvas.copy()
                cv2.rectangle(temp_frame, self.start_pt, self.current_pt, (0, 255, 255), 1)
                cv2.imshow("Device Defect Register", temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                clamped_x = max(0, min(x, self.display_width - 1))
                clamped_y = max(0, min(y, self.display_height - 1))
                self.current_pt = (clamped_x, clamped_y)
                
                w_disp = abs(self.current_pt[0] - self.start_pt[0])
                h_disp = abs(self.current_pt[1] - self.start_pt[1])
                if w_disp < 4 or h_disp < 4:
                    self.redraw_canvas()
                    return

                temp_frame = self.canvas.copy()
                cv2.rectangle(temp_frame, self.start_pt, self.current_pt, (0, 165, 255), 2)
                cv2.putText(temp_frame, "CHOOSE CLASS [1-5] (or Esc to cancel)", 
                            (min(self.start_pt[0], self.current_pt[0]), min(self.start_pt[1], self.current_pt[1]) - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
                
                # Render the prominent modal block warning inside the sidebar panel
                # First, paint a solid dark-gray rectangle over the Y=100 to Y=340 interval to clear previous text
                cv2.rectangle(temp_frame, (self.display_width + 5, 100), (self.display_width + 315, 340), (40, 40, 40), -1)
                
                box_y1 = 105
                box_y2 = 335
                cv2.rectangle(temp_frame, (self.display_width + 5, box_y1), (self.display_width + 315, box_y2), (0, 0, 180), -1) # Dark red fill
                cv2.rectangle(temp_frame, (self.display_width + 5, box_y1), (self.display_width + 315, box_y2), (0, 255, 255), 2)  # Yellow border
                
                cv2.putText(temp_frame, "CHOOSE DEFECT TYPE!", (self.display_width + 15, box_y1 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                
                classes_info = [
                    "[1]: blister",
                    "[2]: tear",
                    "[3]: delamination",
                    "[4]: particulate",
                    "[5]: hole"
                ]
                for idx_cl, c_info in enumerate(classes_info):
                    cy = box_y1 + 55 + idx_cl * 22
                    class_name = ["blister", "tear", "delamination", "particulate", "hole"][idx_cl]
                    cv2.rectangle(temp_frame, (self.display_width + 200, cy - 10), (self.display_width + 215, cy + 2), CLASS_COLORS[class_name], -1)
                    cv2.rectangle(temp_frame, (self.display_width + 200, cy - 10), (self.display_width + 215, cy + 2), (255, 255, 255), 1)
                    cv2.putText(temp_frame, c_info, (self.display_width + 20, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                                
                cv2.putText(temp_frame, "Press [Esc] to cancel box", (self.display_width + 20, box_y1 + 185), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)

                cv2.imshow("Device Defect Register", temp_frame)

                assigned_class = None
                self.is_waiting_for_key = True
                while True:
                    key_press = cv2.waitKeyEx(0) & 0xFF
                    if key_press in KEY_MAPPING:
                        assigned_class = KEY_MAPPING[key_press]
                        break
                    elif key_press == 27: 
                        break
                self.is_waiting_for_key = False

                if assigned_class is not None:
                    scale_up_x = self.native_w / float(self.display_width)
                    scale_up_y = self.native_h / float(self.display_height)

                    orig_x1 = int(round(min(self.start_pt[0], self.current_pt[0]) * scale_up_x))
                    orig_y1 = int(round(min(self.start_pt[1], self.current_pt[1]) * scale_up_y))
                    orig_x2 = int(round(max(self.start_pt[0], self.current_pt[0]) * scale_up_x))
                    orig_y2 = int(round(max(self.start_pt[1], self.current_pt[1]) * scale_up_y))

                    orig_x1 = max(0, min(orig_x1, self.native_w))
                    orig_y1 = max(0, min(orig_y1, self.native_h))
                    orig_x2 = max(0, min(orig_x2, self.native_w))
                    orig_y2 = max(0, min(orig_y2, self.native_h))

                    filename = self.cell_files[self.current_idx]["filename"]
                    cell_data = self.cell_files[self.current_idx]["cell_data"]
                    
                    box_w_px = orig_x2 - orig_x1
                    box_h_px = orig_y2 - orig_y1
                    
                    center_x_px = orig_x1 + (box_w_px / 2.0)
                    center_y_px = orig_y1 + (box_h_px / 2.0)

                    min_x, min_y, max_x, max_y = cell_data["bbox"]
                    
                    gds_w = max_x - min_x
                    gds_h = max_y - min_y
                    
                    center_x_um = min_x + (center_x_px / float(self.native_w)) * gds_w
                    center_y_um = max_y - (center_y_px / float(self.native_h)) * gds_h

                    width_um = (box_w_px / float(self.native_w)) * gds_w
                    height_um = (box_h_px / float(self.native_h)) * gds_h

                    self.annotations[filename].append({
                        "type": assigned_class,
                        "box_px": [orig_x1, orig_y1, box_w_px, box_h_px],
                        "center_x_um": round(center_x_um, 3),
                        "center_y_um": round(center_y_um, 3),
                        "width_um": round(width_um, 3),
                        "height_um": round(height_um, 3)
                    })
                    self.save_annotations_to_file()
                self.redraw_canvas()

    def stitch_and_save_wafer_layout(self):
        """Builds a global composite overview stitch with labeling metadata overlaid."""
        print("\nCompiling GDS-aligned physical wafer overview composite...")
        out_size = self.config.get("output_image_size", 4000)
        composite_canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        half = out_size / 2.0
        scale = (0.925 * half) / self.gds_R

        cv2.circle(composite_canvas, (int(half), int(half)), int(self.gds_R * scale), (60, 60, 60), 2, lineType=cv2.LINE_AA)

        for cell_entry in self.cell_files:
            filename = cell_entry["filename"]
            cell_data = cell_entry["cell_data"]
            min_x, min_y, max_x, max_y = cell_data["bbox"]

            cell_img = cv2.imread(str(cell_entry["filepath"]))
            if cell_img is None: continue

            cell_boxes = self.annotations.get(filename, [])
            for box in cell_boxes:
                box_type = box["type"]
                x, y, w, h = box["box_px"]
                color = CLASS_COLORS.get(box_type, (255, 255, 255))
                cv2.rectangle(cell_img, (x, y), (x + w, y + h), color, 6)
                cv2.putText(cell_img, box_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

            corners_gds = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            pts_img = []
            for gx, gy in corners_gds:
                xi, yi = self.transformer.transform_gds_to_target_img(gx, gy, out_size)
                pts_img.append((xi, yi))
            pts_img = np.array(pts_img)

            tx_min, ty_min = np.min(pts_img, axis=0)
            tx_max, ty_max = np.max(pts_img, axis=0)

            pt_x1 = int(round(tx_min))
            pt_y1 = int(round(ty_min))
            pt_x2 = int(round(tx_max))
            pt_y2 = int(round(ty_max))

            cell_w = pt_x2 - pt_x1
            cell_h = pt_y2 - pt_y1

            if cell_w > 0 and cell_h > 0:
                flipped_cell = cv2.flip(cell_img, 1)
                resized_cell = cv2.resize(flipped_cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                composite_canvas[pt_y1:pt_y2, pt_x1:pt_x2] = resized_cell

        cv2.imwrite(self.output_stitch_path, composite_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Overview composite stitch saved: {self.output_stitch_path}")

    def run(self):
        cv2.namedWindow("Device Defect Register")
        cv2.setMouseCallback("Device Defect Register", self.handle_mouse)
        self.redraw_canvas()
        while True:
            key = cv2.waitKeyEx(20)
            if key in RIGHT_ARROW_CODES or key == 32 or key in [ord('n'), ord('N')]:
                self.save_annotations_to_file()
                if self.current_idx < len(self.cell_files) - 1:
                    self.load_cell_at_index(self.current_idx + 1)
                else: 
                    print("Reached end of device list queue.")
            elif key in LEFT_ARROW_CODES or key in [ord('p'), ord('P')]:
                self.save_annotations_to_file()
                if self.current_idx > 0: 
                    self.load_cell_at_index(self.current_idx - 1)
            elif key == ord('x') or key == ord('X'):
                filename = self.cell_files[self.current_idx]["filename"]
                if filename in self.exclusions:
                    self.exclusions.remove(filename)
                else: 
                    self.exclusions.add(filename)
                self.save_exclusions_file()
                self.redraw_canvas()
            elif key == ord('c') or key == ord('C'):
                filename = self.cell_files[self.current_idx]["filename"]
                self.annotations[filename] = []
                self.redraw_canvas()
                self.save_annotations_to_file()
            elif key == 27 or key in [ord('q'), ord('Q')]:
                self.save_annotations_to_file()
                self.save_exclusions_file()
                self.stitch_and_save_wafer_layout()
                break
        cv2.destroyAllWindows()


# ===========================================================================
# 8. CELL BOUNDS SEGMENTATION AND CLUSTERING
# ===========================================================================

def get_gds_cells_list(polygons, gds_R):
    """Parses design polygons, groups nested shapes, and structures grid rows/cols."""
    raw_cells = []
    for poly in polygons:
        poly_arr = np.array(poly)
        if len(poly_arr) < 3:
            continue
        min_x, min_y = np.min(poly_arr, axis=0)
        max_x, max_y = np.max(poly_arr, axis=0)
        w = max_x - min_x
        h = max_y - min_y
        
        if w > 0.8 * (2 * gds_R) or h > 0.8 * (2 * gds_R):
            continue
        if w < 100.0 or h < 100.0:
            continue
            
        raw_cells.append({
            "bbox": (min_x, min_y, max_x, max_y),
            "center": ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        })

    if not raw_cells:
        return []

    # Merge nested or concentric design masks belonging to the same die structure
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

    print(f"  [GDS Clustering] Merged {len(raw_cells)} design features into {len(merged_cells)} unique physical devices.")

    # Sort centers into coordinate rows and columns
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


# ===========================================================================
# 9. INTEGRATED MAIN WORKFLOW STAGES
# ===========================================================================

def process_wafer_cells(folder, json_file, config, args, wafer_id):
    config_run = copy.deepcopy(config)
    out_stem = wafer_id

    # Overwrite configuration using dynamic summary block parameters if parsed
    if json_file and Path(json_file).exists():
        try:
            defect_data = load_defect_json(json_file)
            summary_block = defect_data.get("summary", {})
            for param in ["overlap_x_percent", "overlap_y_percent", "downscale"]:
                if param in summary_block:
                    val = float(summary_block[param])
                    config_run[param] = val
                    print(f"[{out_stem}] Overriding config using summary json parameter: {param} = {val}")
        except Exception as e:
            print(f"[{out_stem}] Dynamic override parsing warning: {e}")

    print(f"\n[{out_stem}] Detecting camera grid file dimensions...")
    try:
        detected_cols, detected_rows = detect_grid_size(folder)
        config_run["tile_cols"] = detected_cols
        config_run["tile_rows"] = detected_rows
        print(f"[{out_stem}] Dynamic scanning found grid: {detected_cols} columns x {detected_rows} rows.")
    except Exception as e:
        print(f"[{out_stem}] Layout scanning error: {e}")
        return False

    print(f"[{out_stem}] Parsing design metadata from GDS...")
    try:
        gds_xc, gds_yc, gds_R = parse_gds_wafer_boundary(
            config_run["gds_path"],
            layer=config_run["gds_layer"],
            datatype=config_run["gds_datatype"]
        )
        gds_R = float(gds_R)
        print(f"[{out_stem}] GDS Wafer details found: Center=({gds_xc:.1f}, {gds_yc:.1f}), R={gds_R:.1f} um")
        gds_polygons = get_layer0_polygons(config_run["gds_path"])
        print(f"[{out_stem}] Discovered {len(gds_polygons)} features on layer 0.")
    except Exception as e:
        print(f"[{out_stem}] Critical error reading GDS data: {e}")
        return False

    print(f"[{out_stem}] Reassembling downscaled workspace canvas...")
    try:
        ds_canvas, tile_ext = generate_downscaled_stitch(folder, config_run)
    except Exception as e:
        print(f"[{out_stem}] Coarse-stitch canvas generation failed: {e}")
        return False

    ds_factor = config_run["downscale_factor"]
    x_offset_um = 0.0
    y_offset_um = 0.0
    scale_mult = 1.0

    try:
        canvas_xc, canvas_yc, canvas_R, flat_angle = detect_wafer_on_canvas(ds_canvas, ds_factor)

        if args.manual:
            flat_angle, x_offset_um, y_offset_um, scale_mult = run_manual_alignment(
                ds_canvas,
                config_run,
                canvas_xc * ds_factor,
                canvas_yc * ds_factor,
                canvas_R * ds_factor,
                ds_factor,
                tile_ext,
                flat_angle,
                gds_polygons,
                gds_R,
                map_mode=True,
                gds_center=(gds_xc, gds_yc),
                shear=float(config_run.get("shear", 0.0))
            )

        flat_angle = float(flat_angle)
        canvas_xc  = float(canvas_xc)
        canvas_yc  = float(canvas_yc)
        canvas_R   = float(canvas_R)
    except Exception as e:
        print(f"[{out_stem}] Wafer alignment step failed: {e}")
        return False

    exclusions = set()
    exclusions_path = Path("manual_exclusions.json")
    if exclusions_path.exists():
        exclusions = load_exclusions(exclusions_path)

    # Initialize coordinate translation module
    transformer = WaferTransformer(
        canvas_center=(canvas_xc, canvas_yc),
        canvas_radius=canvas_R,
        canvas_flat_angle=flat_angle,
        gds_radius=gds_R,
        config=config_run,
        ext=tile_ext,
        exclusions=exclusions,
        shear=float(config_run.get("shear", 0.0)),
        x_offset=x_offset_um,
        y_offset=y_offset_um,
        map_mode=True,
        gds_center=(gds_xc, gds_yc),
    )

    transformer.S_x *= scale_mult
    transformer.S_y *= scale_mult
    transformer.S   *= scale_mult

    cells = get_gds_cells_list(gds_polygons, gds_R)
    if not cells:
        print(f"[{out_stem}] Critical Error: No device cells identified inside design GDS layer.")
        return False

    run_create = args.create or (not args.create and not args.label)
    run_label = args.label or (not args.create and not args.label)

    out_dir = Path(args.out_dir)

    # --- CELL CROP GENERATION STAGE ---
    if run_create:
        out_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = out_dir / "previews"
        preview_dir.mkdir(exist_ok=True)

        tile_width = config_run["tile_width"]
        tile_height = config_run["tile_height"]
        step_x = tile_width * (1.0 - config_run["overlap_x_percent"] / 100.0)
        step_y = tile_height * (1.0 - config_run["overlap_y_percent"] / 100.0)

        print(f"[{out_stem}] Slicing and rotating {len(cells)} cells at 100% resolution...")
        saved_count = 0
        
        for idx, cell in enumerate(cells):
            row, col = cell["row"], cell["col"]
            min_x, min_y, max_x, max_y = cell["bbox"]

            corners_gds = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            pts_canvas = []
            for gx, gy in corners_gds:
                xc_can, yc_can = transformer.gds_to_canvas(gx, gy)
                pts_canvas.append((xc_can, yc_can))
            pts_canvas = np.array(pts_canvas)

            cx_min, cy_min = np.min(pts_canvas, axis=0)
            cx_max, cy_max = np.max(pts_canvas, axis=0)

            pad = 200
            x1 = int(np.floor(cx_min)) - pad
            y1 = int(np.floor(cy_min)) - pad
            x2 = int(np.ceil(cx_max)) + pad
            y2 = int(np.ceil(cy_max)) + pad

            overlapping_tiles = []
            for c_col in range(1, config_run["tile_cols"] + 1):
                for r_row in range(1, config_run["tile_rows"] + 1):
                    tile_key = f"tile_x{c_col:03d}_y{r_row:03d}{tile_ext}"
                    if tile_key in transformer.exclusions:
                        continue

                    tile_x1 = int(round((c_col - 1) * step_x))
                    tile_y1 = int(round((r_row - 1) * step_y))
                    tile_x2 = tile_x1 + tile_width
                    tile_y2 = tile_y1 + tile_height

                    if max(x1, tile_x1) < min(x2, tile_x2) and max(y1, tile_y1) < min(y2, tile_y2):
                        overlapping_tiles.append((c_col, r_row, tile_x1, tile_y1, tile_x2, tile_y2))

            if not overlapping_tiles:
                continue

            local_w = x2 - x1
            local_h = y2 - y1
            local_canvas = np.zeros((local_h, local_w, 3), dtype=np.uint8)

            for c_col, r_row, tx1, ty1, tx2, ty2 in overlapping_tiles:
                tile_name = f"tile_x{c_col:03d}_y{r_row:03d}{tile_ext}"
                tile_path = Path(folder) / tile_name
                if not tile_path.exists():
                    continue

                tile_img = cv2.imread(str(tile_path))
                if tile_img is None:
                    continue

                loc_tx1 = tx1 - x1
                loc_ty1 = ty1 - y1
                loc_tx2 = tx2 - x1
                loc_ty2 = ty2 - y1

                ox1 = max(0, loc_tx1)
                oy1 = max(0, loc_ty1)
                ox2 = min(local_w, loc_tx2)
                oy2 = min(local_h, loc_ty2)

                sx1 = ox1 - loc_tx1
                sy1 = oy1 - loc_ty1
                sx2 = sx1 + (ox2 - ox1)
                sy2 = sy1 + (oy2 - oy1)

                if (ox2 > ox1) and (oy2 > oy1):
                    local_canvas[oy1:oy2, ox1:ox2] = tile_img[sy1:sy2, sx1:sx2]

            cell_center_canvas = np.mean(pts_canvas, axis=0)
            crop_center = (cell_center_canvas[0] - x1, cell_center_canvas[1] - y1)

            angle_deg = flat_angle * 180.0 / np.pi
            M = cv2.getRotationMatrix2D(crop_center, angle_deg, 1.0)
            rotated_local_canvas = cv2.warpAffine(local_canvas, M, (local_w, local_h), flags=cv2.INTER_LANCZOS4)

            pts_homogeneous = np.column_stack([pts_canvas[:, 0] - x1, pts_canvas[:, 1] - y1, np.ones(4)])
            pts_rotated_local = (M @ pts_homogeneous.T).T

            rx_min, ry_min = np.min(pts_rotated_local, axis=0)
            rx_max, ry_max = np.max(pts_rotated_local, axis=0)

            shave = args.shave
            crop_x1 = int(round(rx_min)) + shave
            crop_x2 = int(round(rx_max)) - shave
            crop_y1 = int(round(ry_min)) + shave
            crop_y2 = int(round(ry_max)) - shave

            crop_x1 = max(0, min(crop_x1, local_w - 1))
            crop_x2 = max(0, min(crop_x2, local_w - 1))
            crop_y1 = max(0, min(crop_y1, local_h - 1))
            crop_y2 = max(0, min(crop_y2, local_h - 1))

            if (crop_x2 > crop_x1) and (crop_y2 > crop_y1):
                cell_crop = rotated_local_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
                cell_crop_gds_perspective = cv2.flip(cell_crop, 1)

                cell_filename = f"{out_stem}_cell_{row}-{col}.jpg"
                out_path = out_dir / cell_filename
                cv2.imwrite(str(out_path), cell_crop_gds_perspective, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Pre-render scaled workspace display copies
                preview_w = 1200
                preview_h = int(preview_w * cell_crop.shape[0] / cell_crop.shape[1])
                cell_preview = cv2.resize(cell_crop_gds_perspective, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                
                preview_filename = f"{out_stem}_cell_{row}-{col}_preview.jpg"
                preview_path = preview_dir / preview_filename
                cv2.imwrite(str(preview_path), cell_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                saved_count += 1
                sys.stdout.write(
                    f"\r[{out_stem} Lossless Crop] Process: {idx+1}/{len(cells)} | "
                    f"Saved cell {row}-{col} ({cell_crop.shape[1]}x{cell_crop.shape[0]} px)\033[K"
                )
                sys.stdout.flush()
                
        print(f"\n[{out_stem}] Completed cell crop generation. Saved {saved_count} native resolutions to: {out_dir}")
    else:
        print(f"[{out_stem}] Skipped cell generation. Using existing files inside folder: {args.out_dir}")

    # --- DEVICE DEFECT INTERACTIVE LABELING STAGE ---
    if args.device and run_label:
        mapper = DeviceDefectMapperTool(
            wafer_id=out_stem,
            cells=cells,
            out_dir=args.out_dir,
            transformer=transformer,
            gds_R=gds_R,
            config=config_run
        )
        mapper.run()

    return True


def parse_batch_file(filepath: str) -> list[dict]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Batch settings text file not found: {filepath}")
        
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        
    wafers = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith(":"):
            wafer_id = line[:-1].strip()
            if i + 3 < len(lines):
                after_folder = lines[i+1].strip('"').strip("'")
                before_folder = lines[i+2].strip('"').strip("'")
                defect_json = lines[i+3].strip('"').strip("'")
                
                if after_folder.endswith(":") or before_folder.endswith(":") or defect_json.endswith(":"):
                    i += 1
                    continue
                
                wafers.append({
                    "id": wafer_id,
                    "after_folder": after_folder,
                    "before_folder": before_folder,
                    "defect_json": defect_json
                })
                i += 4
            else:
                i += 1
        else:
            i += 1
            
    return wafers


# ===========================================================================
# 10. SYSTEM ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone Metrology Core and GDS Extraction Service")
    parser.add_argument("--batch", type=str, required=True,
                        help="Path to wafer batch definitions txt config file")
    parser.add_argument("--manual", action="store_true",
                        help="Launch interactive visualization manual adjustment dashboard beforehand")
    parser.add_argument("--shave", type=int, default=10,
                        help="Crop outer buffer width inside rotated cell frame to erase raster artifacts")
    parser.add_argument("--out-dir", type=str, default="extracted_cells",
                        help="Target output subdirectory to store cropped files")
    parser.add_argument("-d", "--device", action="store_true",
                        help="Enable defect inspection annotation dashboard review")
    parser.add_argument("-c", "--create", action="store_true",
                        help="Stage 1: Generate native cropped die formats")
    parser.add_argument("-l", "--label", action="store_true",
                        help="Stage 2: Label anomalies on extracted images and compile overview layouts")

    args = parser.parse_args()

    try:
        config = load_config("config.json")
    except Exception as e:
        print(f"Error reading local parameters configuration file: {e}")
        sys.exit(1)

    try:
        wafers = parse_batch_file(args.batch)
        print(f"Discovered {len(wafers)} configuration sequences configured in batch run.")
    except Exception as e:
        print(f"Error parsing queue instructions batch file: {e}")
        sys.exit(1)

    for idx, wafer in enumerate(wafers):
        wafer_id = wafer["id"]
        after_folder = wafer["after_folder"]
        defect_json = wafer["defect_json"]

        print("\n" + "=" * 70)
        print(f" WAFER RUN [{idx + 1}/{len(wafers)}]: {wafer_id}")
        print("=" * 70)

        if after_folder.lower() == "none" or not after_folder:
            print(f"[{wafer_id}] Bypassed. Patterned layout directory folder was marked empty.")
            continue

        if not Path(after_folder).exists():
            print(f"[{wafer_id}] Skipping wafer. Target folder directory does not exist: {after_folder}")
            continue

        try:
            process_wafer_cells(
                folder=after_folder,
                json_file=defect_json,
                config=config,
                args=args,
                wafer_id=wafer_id
            )
        except Exception as e:
            print(f"[{wafer_id}] Script crashed during operational pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print(" BATCH EXECUTION RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
