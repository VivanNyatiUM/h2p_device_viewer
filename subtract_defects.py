import os
import json
import math
import argparse
import numpy as np
import gdstk


DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG = 0.002
DEFAULT_ALIGNMENT_ERROR_X_PX = 5.0
DEFAULT_ALIGNMENT_ERROR_Y_PX = 15.0
DEFAULT_EXTRA_MARGIN_UM = 0.0
DEFAULT_WAFER_CENTER_X_UM = 0.0
DEFAULT_WAFER_CENTER_Y_UM = 0.0


def _polygon_signed_area(points: np.ndarray) -> float:
    """Returns signed polygon area. Positive means CCW in normal GDS x/y coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _edge_length(points: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(points[j] - points[i]))


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def estimate_um_per_pixel(defect: dict, corners_gds: np.ndarray) -> tuple[float, float]:
    """
    Estimate native-crop micron-per-pixel in x/y from one defect annotation.

    For new JSONs, corners_gds stores the exact mapped parallelogram for the
    pixel-space box [x, y, w, h]. The GDS length of the top/bottom edges divided
    by |w| gives x um/px; left/right divided by |h| gives y um/px.

    Fallbacks use width_um/height_um for old JSONs, then 1.0 um/px if nothing
    usable exists. Your current crop geometry is roughly ~0.95 um/px, so this is
    sane even when a legacy defect is tiny or malformed.
    """
    px_w = px_h = None
    box_px = defect.get("box_px")
    if isinstance(box_px, (list, tuple)) and len(box_px) >= 4:
        px_w = abs(_safe_float(box_px[2], 0.0) or 0.0)
        px_h = abs(_safe_float(box_px[3], 0.0) or 0.0)

    um_per_px_x = None
    um_per_px_y = None

    if corners_gds is not None and len(corners_gds) >= 4:
        # Expected order from defect_mapper_gui.py:
        # top-left, top-right, bottom-right, bottom-left.
        top = _edge_length(corners_gds, 0, 1)
        right = _edge_length(corners_gds, 1, 2)
        bottom = _edge_length(corners_gds, 2, 3)
        left = _edge_length(corners_gds, 3, 0)

        if px_w and px_w > 1e-9:
            um_per_px_x = 0.5 * (top + bottom) / px_w
        if px_h and px_h > 1e-9:
            um_per_px_y = 0.5 * (right + left) / px_h

    # Legacy/back-compat fallback. These are axis-aligned extents, so less exact,
    # but still better than ignoring the supplied alignment error completely.
    if um_per_px_x is None and px_w and px_w > 1e-9:
        width_um = abs(_safe_float(defect.get("width_um"), 0.0) or 0.0)
        if width_um > 1e-9:
            um_per_px_x = width_um / px_w

    if um_per_px_y is None and px_h and px_h > 1e-9:
        height_um = abs(_safe_float(defect.get("height_um"), 0.0) or 0.0)
        if height_um > 1e-9:
            um_per_px_y = height_um / px_h

    if um_per_px_x is None or not np.isfinite(um_per_px_x) or um_per_px_x <= 0:
        um_per_px_x = 1.0
    if um_per_px_y is None or not np.isfinite(um_per_px_y) or um_per_px_y <= 0:
        um_per_px_y = um_per_px_x

    return float(um_per_px_x), float(um_per_px_y)


def compute_alignment_error_margin_um(
    corners_gds: np.ndarray,
    defect: dict,
    error_angle_deg: float = DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG,
    error_x_px: float = DEFAULT_ALIGNMENT_ERROR_X_PX,
    error_y_px: float = DEFAULT_ALIGNMENT_ERROR_Y_PX,
    extra_margin_um: float = DEFAULT_EXTRA_MARGIN_UM,
    wafer_center: tuple[float, float] = (DEFAULT_WAFER_CENTER_X_UM, DEFAULT_WAFER_CENTER_Y_UM),
) -> float:
    """
    Convert measured alignment uncertainty into one conservative GDS-space margin.

    Components:
      1. x/y registration error in pixels -> microns, using the local annotation's
         own pixel-to-GDS scale.
      2. rotation error in degrees -> microns, using r * dtheta from wafer center.
      3. optional extra fixed margin for process/lithography cushion.

    The returned margin is an isotropic offset distance. It intentionally overcuts
    slightly; that is the point. Better a tiny moat than a surviving defect island.
    """
    um_per_px_x, um_per_px_y = estimate_um_per_pixel(defect, corners_gds)

    # Worst-case translational registration vector in local crop pixels.
    translation_margin_um = math.hypot(
        abs(float(error_x_px)) * um_per_px_x,
        abs(float(error_y_px)) * um_per_px_y,
    )

    # Rotation uncertainty grows with distance from the wafer's rotation center.
    theta = math.radians(abs(float(error_angle_deg)))
    cx, cy = wafer_center
    radii = np.linalg.norm(corners_gds - np.array([[float(cx), float(cy)]], dtype=np.float64), axis=1)
    max_radius_um = float(np.max(radii)) if len(radii) else 0.0
    rotation_margin_um = max_radius_um * math.sin(theta)

    margin = translation_margin_um + rotation_margin_um + max(0.0, float(extra_margin_um))
    return float(max(0.0, margin))


def _scale_polygon_about_centroid(points: np.ndarray, margin_um: float) -> np.ndarray:
    """
    Fallback expansion if gdstk.offset ever fails on a degenerate polygon.
    This scales vertices away from the centroid. gdstk.offset is preferred.
    """
    centroid = np.mean(points, axis=0)
    vectors = points - centroid
    max_radius = float(np.max(np.linalg.norm(vectors, axis=1)))
    if max_radius <= 1e-9:
        return points.copy()
    scale = (max_radius + margin_um) / max_radius
    return centroid + vectors * scale


def expand_defect_polygon_for_alignment_error(
    points: np.ndarray,
    margin_um: float,
    precision: float = 1e-3,
) -> list[gdstk.Polygon]:
    """
    Expand a defect polygon outward by a fixed GDS-space margin.

    This is the actual error-compensation function. It takes the coordinates of
    the defect polygon and returns one or more polygons enlarged enough to cover
    the measured alignment uncertainty.
    """
    if margin_um <= 0:
        return [gdstk.Polygon(points)]

    base_poly = gdstk.Polygon(points)
    try:
        expanded = gdstk.offset(
            [base_poly],
            margin_um,
            join="miter",
            tolerance=2,
            precision=precision,
            use_union=True,
        )
        if expanded:
            return expanded
    except Exception as e:
        print(f"[WARN] gdstk.offset failed on a defect polygon; using centroid scaling fallback: {e}")

    return [gdstk.Polygon(_scale_polygon_about_centroid(points, margin_um))]


def _legacy_axis_aligned_points(defect: dict) -> np.ndarray:
    """Rebuild old center/width/height annotations as an axis-aligned rectangle."""
    cx = float(defect["center_x_um"])
    cy = float(defect["center_y_um"])
    w = float(defect["width_um"])
    h = float(defect["height_um"])
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float64)


def _load_defect_polygons(
    defects_data: dict,
    compensate_alignment_error: bool = True,
    error_angle_deg: float = DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG,
    error_x_px: float = DEFAULT_ALIGNMENT_ERROR_X_PX,
    error_y_px: float = DEFAULT_ALIGNMENT_ERROR_Y_PX,
    extra_margin_um: float = DEFAULT_EXTRA_MARGIN_UM,
    wafer_center: tuple[float, float] = (DEFAULT_WAFER_CENTER_X_UM, DEFAULT_WAFER_CENTER_Y_UM),
    precision: float = 1e-3,
) -> tuple[list[gdstk.Polygon], int, int, list[float]]:
    """Load JSON defects into GDS polygons, optionally expanding for alignment error."""
    defect_polygons = []
    legacy_count = 0
    rotated_count = 0
    margins = []

    for filename, defects in defects_data.items():
        # Skip metadata blocks if they ever appear in the JSON.
        if not isinstance(defects, list):
            continue

        for defect in defects:
            if not isinstance(defect, dict):
                continue

            if "corners_gds" in defect and defect["corners_gds"]:
                points = np.array([(float(x), float(y)) for x, y in defect["corners_gds"]], dtype=np.float64)
                rotated_count += 1
            else:
                points = _legacy_axis_aligned_points(defect)
                legacy_count += 1

            if len(points) < 3:
                print(f"[WARN] Skipping malformed defect in {filename}: fewer than 3 polygon points.")
                continue

            # Remove repeated closing point if present.
            if len(points) > 3 and np.linalg.norm(points[0] - points[-1]) < 1e-9:
                points = points[:-1]

            if abs(_polygon_signed_area(points)) < 1e-9:
                print(f"[WARN] Skipping degenerate zero-area defect polygon in {filename}.")
                continue

            if compensate_alignment_error:
                margin_um = compute_alignment_error_margin_um(
                    points,
                    defect,
                    error_angle_deg=error_angle_deg,
                    error_x_px=error_x_px,
                    error_y_px=error_y_px,
                    extra_margin_um=extra_margin_um,
                    wafer_center=wafer_center,
                )
                margins.append(margin_um)
                defect_polygons.extend(
                    expand_defect_polygon_for_alignment_error(points, margin_um, precision=precision)
                )
            else:
                defect_polygons.append(gdstk.Polygon(points))

    return defect_polygons, legacy_count, rotated_count, margins


def subtract_defects_from_gds(
    gds_path,
    json_path,
    output_path,
    target_layers=[4],
    compensate_alignment_error: bool = True,
    error_angle_deg: float = DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG,
    error_x_px: float = DEFAULT_ALIGNMENT_ERROR_X_PX,
    error_y_px: float = DEFAULT_ALIGNMENT_ERROR_Y_PX,
    extra_margin_um: float = DEFAULT_EXTRA_MARGIN_UM,
    wafer_center: tuple[float, float] = (DEFAULT_WAFER_CENTER_X_UM, DEFAULT_WAFER_CENTER_Y_UM),
    strict_corners: bool = False,
    precision: float = 1e-3,
):
    """
    Recursively flattens specified layers in a GDSII hierarchy and subtracts
    defect regions, preserving other layers intact.

    IMPORTANT: defect regions should be subtracted as the exact rotated
    quadrilateral recorded by defect_mapper_gui.py ("corners_gds"), NOT as an
    axis-aligned box rebuilt from center/width/height. The native crop pixel
    frame is rotated relative to the GDS axes by the wafer's flat angle, so an
    axis-aligned pixel box maps to a rotated parallelogram in GDS microns.

    New in this version:
      Defect polygons are conservatively expanded to absorb alignment error.
      Defaults match your measured worst case:
        angle ~= 0.002 degrees, x ~= 5 px, y ~= 15 px.
    """
    if not os.path.exists(gds_path):
        raise FileNotFoundError(f"GDS file not found: {gds_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Defect JSON file not found: {json_path}")

    # 1. Load defect annotations from JSON
    with open(json_path, "r") as f:
        defects_data = json.load(f)

    defect_polygons, legacy_count, rotated_count, margins = _load_defect_polygons(
        defects_data,
        compensate_alignment_error=compensate_alignment_error,
        error_angle_deg=error_angle_deg,
        error_x_px=error_x_px,
        error_y_px=error_y_px,
        extra_margin_um=extra_margin_um,
        wafer_center=wafer_center,
        precision=precision,
    )

    if legacy_count > 0:
        msg = (f"{legacy_count} defect(s) had no 'corners_gds' field and were reconstructed as "
               f"axis-aligned boxes from center/width/height. These can still be misaligned "
               f"whenever the wafer flat angle is nonzero. Re-annotate or migrate the JSON "
               f"to regenerate corners_gds for exact subtraction geometry.")
        if strict_corners:
            raise RuntimeError(msg)
        print(f"[WARNING] {msg}")

    print(f"[INFO] {rotated_count} defect(s) loaded using exact rotated GDS corners.")

    if compensate_alignment_error:
        print(f"[INFO] Alignment-error compensation enabled: angle={error_angle_deg:.6f}°, "
              f"x={error_x_px:.3f}px, y={error_y_px:.3f}px, extra={extra_margin_um:.3f} µm")
        if margins:
            arr = np.array(margins, dtype=np.float64)
            print(f"[INFO] Defect expansion margin: min={arr.min():.3f} µm, "
                  f"mean={arr.mean():.3f} µm, max={arr.max():.3f} µm")
    else:
        print("[INFO] Alignment-error compensation disabled.")

    if not defect_polygons:
        print("[INFO] No defects discovered in JSON. Writing original file unchanged.")
        lib = gdstk.read_gds(gds_path)
        lib.write_gds(output_path)
        return

    # 2. Read the original GDS fully (no layer filtering)
    print(f"[INFO] Reading full GDS: {gds_path}...")
    lib = gdstk.read_gds(gds_path)
    top_cells = lib.top_level()
    if not top_cells:
        raise ValueError("No top-level cells found in GDS.")
    original_top = top_cells[0]

    # 3. Recursively extract ALL polygons & convert paths down the entire hierarchy
    print("[INFO] Recursively traversing design hierarchy and converting paths...")
    all_polygons = original_top.get_polygons(apply_repetitions=True)
    print(f"[INFO] Discovered {len(all_polygons)} total polygons across all layers.")

    # 4. Group all extracted polygons by (layer, datatype) to preserve design metadata
    polys_by_layer_type = {}
    for poly in all_polygons:
        key = (poly.layer, poly.datatype)
        if key not in polys_by_layer_type:
            polys_by_layer_type[key] = []
        polys_by_layer_type[key].append(poly)

    # 5. Perform boolean subtraction on specified layers
    final_polygons = []
    target_layers = set(int(layer) for layer in target_layers)

    for (layer, datatype), layer_polys in polys_by_layer_type.items():
        if layer in target_layers:
            print(f"  -> Cutting defect regions from Layer {layer}, Datatype {datatype} ({len(layer_polys)} polys)...")
            subtracted = gdstk.boolean(
                layer_polys,
                defect_polygons,
                "not",
                precision=precision,
                layer=layer,
                datatype=datatype,
            )
            final_polygons.extend(subtracted)
        else:
            # Keep other layers completely untouched
            final_polygons.extend(layer_polys)

    # 6. Create a clean, flat output cell
    new_top_cell = gdstk.Cell(original_top.name)
    for p in final_polygons:
        new_top_cell.add(p)

    output_lib = gdstk.Library(name=lib.name, unit=lib.unit, precision=lib.precision)
    output_lib.add(new_top_cell)
    output_lib.write_gds(output_path)
    print(f"[SUCCESS] Subtraction complete! Saved flat GDS file to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtract defect regions from specified layers in a GDS.")
    parser.add_argument("--gds", type=str, default="semiconductor_design.gds", help="Path to original GDS")
    parser.add_argument("--json", type=str, required=True, help="Path to defect JSON data")
    parser.add_argument("--out", type=str, required=True, help="Path to write modified output GDS")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4], help="Target GDS layers for boolean subtraction, e.g. --layers 1 2 4")

    parser.add_argument("--error-angle-deg", type=float, default=DEFAULT_ALIGNMENT_ERROR_ANGLE_DEG, help="Worst-case residual rotation error in degrees; default: 0.002")
    parser.add_argument("--error-x-px", type=float, default=DEFAULT_ALIGNMENT_ERROR_X_PX, help="Worst-case residual x registration error in native crop pixels; default: 5")
    parser.add_argument("--error-y-px", type=float, default=DEFAULT_ALIGNMENT_ERROR_Y_PX, help="Worst-case residual y registration error in native crop pixels; default: 15")
    parser.add_argument("--extra-margin-um", type=float, default=DEFAULT_EXTRA_MARGIN_UM, help="Additional fixed safety margin in GDS microns; default: 0")
    parser.add_argument("--wafer-center-x-um", type=float, default=DEFAULT_WAFER_CENTER_X_UM, help="Wafer rotation center X in GDS microns for angle-error compensation; default: 0")
    parser.add_argument("--wafer-center-y-um", type=float, default=DEFAULT_WAFER_CENTER_Y_UM, help="Wafer rotation center Y in GDS microns for angle-error compensation; default: 0")
    parser.add_argument("--no-error-compensation", action="store_true", help="Disable conservative polygon expansion for alignment uncertainty")
    parser.add_argument("--strict-corners", action="store_true", help="Fail instead of falling back when any defect is missing corners_gds")
    parser.add_argument("--precision", type=float, default=1e-3, help="GDS boolean/offset precision in microns; default: 1e-3")

    args = parser.parse_args()

    subtract_defects_from_gds(
        args.gds,
        args.json,
        args.out,
        args.layers,
        compensate_alignment_error=not args.no_error_compensation,
        error_angle_deg=args.error_angle_deg,
        error_x_px=args.error_x_px,
        error_y_px=args.error_y_px,
        extra_margin_um=args.extra_margin_um,
        wafer_center=(args.wafer_center_x_um, args.wafer_center_y_um),
        strict_corners=args.strict_corners,
        precision=args.precision,
    )
