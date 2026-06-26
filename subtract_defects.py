import os
import json
import argparse
import numpy as np
import gdstk

def subtract_defects_from_gds(gds_path, json_path, output_path, target_layers=[4]):
    """
    Recursively flattens specified layers in a GDSII hierarchy and 
    subtracts defect rectangles, preserving other layers intact.
    """
    if not os.path.exists(gds_path):
        raise FileNotFoundError(f"GDS file not found: {gds_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Defect JSON file not found: {json_path}")

    # 1. Load defect annotations from JSON
    with open(json_path, "r") as f:
        defects_data = json.load(f)

    # Convert center/width/height in microns to bounding box boundaries (x1, y1, x2, y2)
    defect_boxes = []
    for filename, defects in defects_data.items():
        for defect in defects:
            cx = float(defect["center_x_um"])
            cy = float(defect["center_y_um"])
            w = float(defect["width_um"])
            h = float(defect["height_um"])
            
            x1, y1 = cx - w / 2.0, cy - h / 2.0
            x2, y2 = cx + w / 2.0, cy + h / 2.0
            defect_boxes.append((x1, y1, x2, y2))

    if not defect_boxes:
        print("[INFO] No defects discovered in JSON. Writing original file unchanged.")
        lib = gdstk.read_gds(gds_path)
        lib.write_gds(output_path)
        return

    # Convert boundaries to gdstk rectangle objects
    defect_polygons = [gdstk.rectangle((b[0], b[1]), (b[2], b[3])) for b in defect_boxes]

    # 2. Read the original GDS fully (no layer filtering)
    print(f"[INFO] Reading full GDS: {gds_path}...")
    lib = gdstk.read_gds(gds_path)
    original_top = lib.top_level()[0]

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
    for (layer, datatype), layer_polys in polys_by_layer_type.items():
        if layer in target_layers:
            print(f"  -> Cutting defect regions from Layer {layer}, Datatype {datatype} ({len(layer_polys)} polys)...")
            # Subtract defect polygons from this layer's geometries
            subtracted = gdstk.boolean(
                layer_polys, defect_polygons, "not", 
                precision=1e-3, layer=layer, datatype=datatype
            )
            final_polygons.extend(subtracted)
        else:
            # Keep other layers completely untouched
            final_polygons.extend(layer_polys)

    # 6. Create a clean, flat output cell with no duplicates
    new_top_cell = gdstk.Cell(original_top.name)
    for p in final_polygons:
        new_top_cell.add(p)

    # Write out a clean library containing only the modified flat top cell
    output_lib = gdstk.Library(name=lib.name, unit=lib.unit, precision=lib.precision)
    output_lib.add(new_top_cell)
    output_lib.write_gds(output_path)
    
    print(f"[SUCCESS] Subtraction complete! Saved flat GDS file to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtract defect boxes from specified layers in a GDS.")
    parser.add_argument("--gds", type=str, default="semiconductor_design.gds", help="Path to original GDS")
    parser.add_argument("--json", type=str, required=True, help="Path to defect JSON data")
    parser.add_argument("--out", type=str, required=True, help="Path to write modified output GDS")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4], help="Target GDS layers for boolean subtraction (e.g. --layers 1 2 4)")
    
    args = parser.parse_args()
    subtract_defects_from_gds(args.gds, args.json, args.out, args.layers)