import sys
import os
import json
import copy
import math
import argparse
import re
from pathlib import Path
import numpy as np
import cv2

# Import project modules
import coordinate_transformer
import gds_parser
import wafer_metrology
import wafer_align_gui
import defect_mapper_gui
import large_wafer_tester
import centroid_algorithm

# ===========================================================================
# 1. IO METADATA UTILITIES
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


def detect_grid_size(tile_folder):
    folder = Path(tile_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Tile folder does not exist at: {tile_folder}")
    pattern = re.compile(r'tile_x(\d+)_y(\d+)')
    max_col = max_row = 0
    for file in folder.iterdir():
        if file.is_file():
            match = pattern.search(file.name)
            if match:
                col, row = int(match.group(1)), int(match.group(2))
                max_col, max_row = max(max_col, col), max(max_row, row)
    if max_col == 0 or max_row == 0:
        raise ValueError(f"No valid tile files matched inside folder: {tile_folder}")
    return max_col, max_row


def load_exclusions(exclusions_path):
    path = Path(exclusions_path)
    if not path.exists():
        return set()
    try:
        with open(path, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


# ===========================================================================
# 2. CORE WAFER PROCESSING PIPELINE
# ===========================================================================

def process_wafer_cells(folder, json_file, config, args, wafer_id):
    config_run = copy.deepcopy(config)
    out_stem = wafer_id

    if json_file and Path(json_file).exists():
        try:
            defect_data = load_defect_json(json_file)
            summary_block = defect_data.get("summary", {})
            for param in ["overlap_x_percent", "overlap_y_percent", "downscale"]:
                if param in summary_block:
                    config_run[param] = float(summary_block[param])
        except Exception as e:
            print(f"[{out_stem}] Dynamic override parsing warning: {e}")

    try:
        detected_cols, detected_rows = detect_grid_size(folder)
        config_run["tile_cols"], config_run["tile_rows"] = detected_cols, detected_rows
    except Exception as e:
        print(f"[{out_stem}] Layout scanning error: {e}")
        return False

    try:
        gds_xc, gds_yc, gds_R = gds_parser.parse_gds_wafer_boundary(
            config_run["gds_path"], layer=config_run.get("gds_layer", 2), datatype=config_run.get("gds_datatype", 0)
        )
        gds_R = float(gds_R)
        gds_polygons = gds_parser.get_gds_overlay_polygons(config_run["gds_path"], config_run)
    except Exception as e:
        print(f"[{out_stem}] Critical error reading GDS data: {e}")
        return False

    try:
        ds_canvas, tile_ext = wafer_metrology.generate_downscaled_stitch(folder, config_run)
    except Exception as e:
        print(f"[{out_stem}] Coarse-stitch canvas generation failed: {e}")
        return False

    ds_factor = config_run["downscale_factor"]
    x_offset_um, y_offset_um, scale_mult = 0.0, 0.0, 1.0

    try:
        canvas_xc, canvas_yc, canvas_R, flat_angle = wafer_metrology.detect_wafer_on_canvas(ds_canvas, ds_factor)
        markers = gds_parser.parse_alignment_markers(config_run["gds_path"])

        if args.manual:
            print(f"\n[{out_stem}] Launching automated Centroid Snapping UI on tiles...")
            try:
                tester = large_wafer_tester.LargeWaferTester(image_path=folder, display_height=800, debug=args.centroid_debug)
                tester.run()
                
                phys_list, nom_list = [], []

                # Terminology updated: `"circle"` is now correctly renamed to `"square"`
                left_squares = [m for m in markers["left"] if m["type"] == "square"]
                right_squares = [m for m in markers["right"] if m["type"] == "square"]

                left_gds_cx = np.mean([m["center"][0] for m in left_squares]) if left_squares else np.mean([m["center"][0] for m in markers["left"]])
                left_gds_cy = np.mean([m["center"][1] for m in left_squares]) if left_squares else np.mean([m["center"][1] for m in markers["left"]])
                right_gds_cx = np.mean([m["center"][0] for m in right_squares]) if right_squares else np.mean([m["center"][0] for m in markers["right"]])
                right_gds_cy = np.mean([m["center"][1] for m in right_squares]) if right_squares else np.mean([m["center"][1] for m in markers["right"]])

                # Match Left Marker corners directly to absolute CAD Bounding Boxes (prevents horizontal offsets)
                if tester.left_boxes_global:
                    for (row, col), corners in tester.left_boxes_global.items():
                        if len(corners) == 4:
                            target_x = left_gds_cx + centroid_algorithm.NOMINAL_COORDS[(row, col)][0]
                            target_y = left_gds_cy - centroid_algorithm.NOMINAL_COORDS[(row, col)][1]
                            
                            best_match = None
                            min_dist = float("inf")
                            for m in left_squares:
                                dist = math.hypot(m["center"][0] - target_x, m["center"][1] - target_y)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match = m
                                    
                            if best_match is not None and min_dist < 200.0:
                                min_x, min_y, max_x, max_y = best_match["bbox"]
                                nom_corners = [
                                    (min_x, max_y),  # 0: Top-Left
                                    (max_x, max_y),  # 1: Top-Right
                                    (max_x, min_y),  # 2: Bottom-Right
                                    (min_x, min_y)   # 3: Bottom-Left
                                ]
                                for i in range(4):
                                    phys_list.append(corners[i])
                                    nom_list.append(nom_corners[i])

                # Match Right Marker corners directly to absolute CAD Bounding Boxes (prevents horizontal offsets)
                if tester.right_boxes_global:
                    for (row, col), corners in tester.right_boxes_global.items():
                        if len(corners) == 4:
                            target_x = right_gds_cx + centroid_algorithm.NOMINAL_COORDS[(row, col)][0]
                            target_y = right_gds_cy - centroid_algorithm.NOMINAL_COORDS[(row, col)][1]
                            
                            best_match = None
                            min_dist = float("inf")
                            for m in right_squares:
                                dist = math.hypot(m["center"][0] - target_x, m["center"][1] - target_y)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match = m
                                    
                            if best_match is not None and min_dist < 200.0:
                                min_x, min_y, max_x, max_y = best_match["bbox"]
                                nom_corners = [
                                    (min_x, max_y),  # 0: Top-Left
                                    (max_x, max_y),  # 1: Top-Right
                                    (max_x, min_y),  # 2: Bottom-Right
                                    (min_x, min_y)   # 3: Bottom-Left
                                ]
                                for i in range(4):
                                    phys_list.append(corners[i])
                                    nom_list.append(nom_corners[i])

                if len(phys_list) >= 4:
                    phys_arr, nom_arr = np.array(phys_list), np.array(nom_list)
                    S_0 = gds_R / canvas_R
                    
                    # Convert coordinates relative to wafer center (with physical Y inverted)
                    x_cart = (phys_arr[:, 0] - canvas_xc) * S_0
                    y_cart = (canvas_yc - phys_arr[:, 1]) * S_0
                    base_cart_arr = np.column_stack((x_cart, y_cart))

                    scale_mult, R_mat, t_vec, rmsd = coordinate_transformer.umeyama_rigid_registration(base_cart_arr, nom_arr)
                    
                    # SVD rotation provides absolute global flat angle of the wafer
                    flat_angle = math.atan2(R_mat[1, 0], R_mat[0, 0])
                    
                    # Normalized translation relative to absolute GDS wafer center coordinate
                    x_offset_um = float(t_vec[0, 0]) - gds_xc
                    y_offset_um = float(t_vec[1, 0]) - gds_yc
                    
                    print(f"[{out_stem} Auto-Align] Global Rigid SVD registration complete:")
                    print(f"  RMSD: {rmsd:.3f} um over {len(phys_arr)} point pairs")
                    print(f"  Solved Flat Angle: {flat_angle * 180 / np.pi:.4f}°")
                    print(f"  Solved Scale Multiplier: {scale_mult:.6f}")
                    print(f"  Solved Translation: X={x_offset_um:.1f} um, Y={y_offset_um:.1f} um")
                else:
                    print(f"[{out_stem} Auto-Align] Warning: Not enough points resolved for SVD pre-alignment.")

            except Exception as e:
                print(f"[{out_stem} Auto-Align] Warning: SVD alignment calculation bypassed ({e}). Using metrology defaults.")

            # Load manual calibration with auto-preset parameters populated inside slider fields
            flat_angle, x_offset_um, y_offset_um, scale_mult = wafer_align_gui.run_manual_alignment(
                ds_canvas, config_run, canvas_xc * ds_factor, canvas_yc * ds_factor, canvas_R * ds_factor,
                ds_factor, tile_ext, flat_angle, gds_polygons, gds_R, map_mode=True, gds_center=(gds_xc, gds_yc),
                shear=float(config_run.get("shear", 0.0)), markers=markers, initial_tx=-x_offset_um, initial_ty=-y_offset_um, initial_scale=scale_mult
            )

        flat_angle, canvas_xc, canvas_yc, canvas_R = float(flat_angle), float(canvas_xc), float(canvas_yc), float(canvas_R)
    except Exception as e:
        print(f"[{out_stem}] Wafer metrology alignment failed: {e}")
        return False

    exclusions = load_exclusions("manual_exclusions.json")
    transformer = coordinate_transformer.WaferTransformer(
        canvas_center=(canvas_xc, canvas_yc), canvas_radius=canvas_R, canvas_flat_angle=flat_angle, gds_radius=gds_R,
        config=config_run, ext=tile_ext, exclusions=exclusions, shear=float(config_run.get("shear", 0.0)),
        x_offset=x_offset_um, y_offset=y_offset_um, map_mode=True, gds_center=(gds_xc, gds_yc)
    )
    transformer.S_x, transformer.S_y, transformer.S = transformer.S_x * scale_mult, transformer.S_y * scale_mult, transformer.S * scale_mult

    cells = gds_parser.get_gds_cells_list(gds_polygons, gds_R)
    if not cells:
        print(f"[{out_stem}] Critical Error: No device cells identified inside design GDS layer.")
        return False

    run_create = args.create or (not args.create and not args.label)
    run_label = args.label or (not args.create and not args.label)
    out_dir = Path(args.out_dir)

    # --- NATIVE DIES CROP EXTRACTION PASS ---
    if run_create:
        out_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = out_dir / "previews"
        preview_dir.mkdir(exist_ok=True)

        tile_width, tile_height = config_run["tile_width"], config_run["tile_height"]
        step_x = tile_width * (1.0 - config_run["overlap_x_percent"] / 100.0)
        step_y = tile_height * (1.0 - config_run["overlap_y_percent"] / 100.0)

        saved_count = 0
        for idx, cell in enumerate(cells):
            row, col = cell["row"], cell["col"]
            min_x, min_y, max_x, max_y = cell["bbox"]
            pts_canvas = np.array([transformer.gds_to_canvas(gx, gy) for gx, gy in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]])

            cx_min, cy_min = np.min(pts_canvas, axis=0)
            cx_max, cy_max = np.max(pts_canvas, axis=0)
            pad = 200
            x1, y1 = int(np.floor(cx_min)) - pad, int(np.floor(cy_min)) - pad
            x2, y2 = int(np.ceil(cx_max)) + pad, int(np.ceil(cy_max)) + pad

            overlapping_tiles = []
            for c_col in range(1, config_run["tile_cols"] + 1):
                for r_row in range(1, config_run["tile_rows"] + 1):
                    tile_key = f"tile_x{c_col:03d}_y{r_row:03d}{tile_ext}"
                    if tile_key in transformer.exclusions:
                        continue
                    tile_x1, tile_y1 = int(round((c_col - 1) * step_x)), int(round((r_row - 1) * step_y))
                    tile_x2, tile_y2 = tile_x1 + tile_width, tile_y1 + tile_height
                    if max(x1, tile_x1) < min(x2, tile_x2) and max(y1, tile_y1) < min(y2, tile_y2):
                        overlapping_tiles.append((c_col, r_row, tile_x1, tile_y1, tile_x2, tile_y2))

            if not overlapping_tiles:
                continue

            local_w, local_h = x2 - x1, y2 - y1
            local_canvas = np.zeros((local_h, local_w, 3), dtype=np.uint8)

            for c_col, r_row, tx1, ty1, tx2, ty2 in overlapping_tiles:
                tile_path = Path(folder) / f"tile_x{c_col:03d}_y{r_row:03d}{tile_ext}"
                if not tile_path.exists():
                    continue
                tile_img = cv2.imread(str(tile_path))
                if tile_img is None:
                    continue

                loc_tx1, loc_ty1 = tx1 - x1, ty1 - y1
                loc_tx2, loc_ty2 = tx2 - x1, ty2 - y1
                ox1, oy1 = max(0, loc_tx1), max(0, loc_ty1)
                ox2, oy2 = min(local_w, loc_tx2), min(local_h, loc_ty2)
                sx1, sy1 = ox1 - loc_tx1, oy1 - loc_ty1
                sx2, sy2 = sx1 + (ox2 - ox1), sy1 + (oy2 - oy1)

                if (ox2 > ox1) and (oy2 > oy1):
                    local_canvas[oy1:oy2, ox1:ox2] = tile_img[sy1:sy2, sx1:sx2]

            crop_center = (np.mean(pts_canvas, axis=0)[0] - x1, np.mean(pts_canvas, axis=0)[1] - y1)
            M = cv2.getRotationMatrix2D(crop_center, flat_angle * 180.0 / np.pi, 1.0)
            rotated_local_canvas = cv2.warpAffine(local_canvas, M, (local_w, local_h), flags=cv2.INTER_LINEAR)

            pts_rotated_local = (M @ np.column_stack([pts_canvas[:, 0] - x1, pts_canvas[:, 1] - y1, np.ones(4)]).T).T
            rx_min, ry_min = np.min(pts_rotated_local, axis=0)
            rx_max, ry_max = np.max(pts_rotated_local, axis=0)

            shave = args.shave
            crop_x1 = max(0, min(int(round(rx_min)) + shave, local_w - 1))
            crop_x2 = max(0, min(int(round(rx_max)) - shave, local_w - 1))
            crop_y1 = max(0, min(int(round(ry_min)) + shave, local_h - 1))
            crop_y2 = max(0, min(int(round(ry_max)) - shave, local_h - 1))

            if (crop_x2 > crop_x1) and (crop_y2 > crop_y1):
                cell_crop = rotated_local_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
                cv2.imwrite(str(out_dir / f"{out_stem}_cell_{row}-{col}.jpg"), cell_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                cell_preview = cv2.resize(cell_crop, (1200, int(1200 * cell_crop.shape[0] / cell_crop.shape[1])), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(preview_dir / f"{out_stem}_cell_{row}-{col}_preview.jpg"), cell_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved_count += 1
                sys.stdout.write(f"\r[{out_stem} Lossless Crop] Process: {idx+1}/{len(cells)} | Saved cell {row}-{col}\033[K")
                sys.stdout.flush()
        print(f"\n[{out_stem}] Slicing complete. Extracted {saved_count} cells.")
    else:
        print(f"[{out_stem}] Slicing skipped. Target cells loaded directly.")

    # --- INTERACTIVE DEFECT ANNOTATION REVIEW PASS ---
    if args.device and run_label:
        mapper = defect_mapper_gui.DeviceDefectMapperTool(
            wafer_id=out_stem, cells=cells, out_dir=args.out_dir, transformer=transformer,
            gds_R=gds_R, config=config_run, shave=args.shave, pad=200
        )
        mapper.run()

    return True


def parse_batch_file(filepath: str) -> list[dict]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Batch definitions config file not found: {filepath}")
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
                wafers.append({"id": wafer_id, "after_folder": after_folder, "before_folder": before_folder, "defect_json": defect_json})
                i += 4
            else:
                i += 1
        else:
            i += 1
    return wafers


# ===========================================================================
# 3. SYSTEM LAUNCH ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone Metrology Core and GDS Extraction Service")
    parser.add_argument("--batch", type=str, required=True, help="Path to wafer batch definitions txt config file")
    parser.add_argument("--manual", action="store_true", help="Launch interactive manual adjustment dashboard beforehand")
    parser.add_argument("--shave", type=int, default=10, help="Crop outer buffer width inside rotated cell frame")
    parser.add_argument("--out-dir", type=str, default="extracted_cells", help="Target output subdirectory to store cropped files")
    parser.add_argument("-d", "--device", action="store_true", help="Enable defect inspection annotation dashboard review")
    parser.add_argument("-c", "--create", action="store_true", help="Stage 1: Generate native cropped die formats")
    parser.add_argument("-l", "--label", action="store_true", help="Stage 2: Label anomalies on extracted images")
    parser.add_argument("--centroid-debug", action="store_true", help="Enable debug mode in automated alignment")

    args = parser.parse_args()

    try:
        config = load_config("config.json")
    except Exception as e:
        print(f"Error reading parameters config file: {e}")
        sys.exit(1)

    try:
        wafers = parse_batch_file(args.batch)
        print(f"Discovered {len(wafers)} configuration sequences configured in batch run.")
    except Exception as e:
        print(f"Error parsing instruction batch file: {e}")
        sys.exit(1)

    for idx, wafer in enumerate(wafers):
        wafer_id = wafer["id"]
        after_folder = wafer["after_folder"]
        defect_json = wafer["defect_json"]

        print("\n" + "=" * 70)
        print(f" WAFER RUN [{idx + 1}/{len(wafers)}]: {wafer_id}")
        print("=" * 70)

        if after_folder.lower() == "none" or not after_folder:
            continue
        if not Path(after_folder).exists():
            continue

        try:
            process_wafer_cells(folder=after_folder, json_file=defect_json, config=config, args=args, wafer_id=wafer_id)
        except Exception as e:
            print(f"[{wafer_id}] Script crashed during operational pipeline execution: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(" BATCH EXECUTION RUN COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()