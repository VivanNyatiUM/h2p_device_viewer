import os
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

CLASS_COLORS = {
    "blister": (0, 255, 0),
    "tear": (255, 0, 0),
    "delamination": (255, 0, 255),
    "particulate": (0, 0, 255),
    "hole": (0, 255, 255)
}

KEY_MAPPING = {ord('1'): "blister", ord('2'): "tear", ord('3'): "delamination", ord('4'): "particulate", ord('5'): "hole"}

LEFT_ARROW_CODES = [2424832, 65361, 81, 0x250000, 63234]
RIGHT_ARROW_CODES = [2555904, 65363, 83, 0x270000, 63235]


class DeviceDefectMapperTool:
    def __init__(self, wafer_id, cells, out_dir, transformer, gds_R, config, shave: int = 10, pad: int = 200):
        self.wafer_id, self.cells, self.out_dir = wafer_id, cells, Path(out_dir)
        self.transformer, self.gds_R, self.config = transformer, gds_R, config
        self.max_disp_w, self.max_disp_h = 950, 750

        # Store shave and pad so every crop_pixel_to_gds call uses values that
        # exactly match the Stage 1 extraction pass.
        self.shave = shave
        self.pad = pad

        self.output_json_path = f"{wafer_id}_device_defects.json"
        self.output_stitch_path = f"{wafer_id}_stitched_devices.jpg"

        self.cell_files = []
        for cell in self.cells:
            filepath = self.out_dir / f"{wafer_id}_cell_{cell['row']}-{cell['col']}.jpg"
            if filepath.exists():
                self.cell_files.append({"filename": filepath.name, "filepath": filepath, "cell_data": cell})

        if not self.cell_files:
            raise FileNotFoundError(f"No cell crops discovered in '{self.out_dir}'. Ensure Stage 1 extraction succeeded first.")

        self.cell_files.sort(key=lambda x: (x["cell_data"]["row"], x["cell_data"]["col"]))

        self.all_min_x = min(c["cell_data"]["bbox"][0] for c in self.cell_files)
        self.all_min_y = min(c["cell_data"]["bbox"][1] for c in self.cell_files)
        self.all_max_x = max(c["cell_data"]["bbox"][2] for c in self.cell_files)
        self.all_max_y = max(c["cell_data"]["bbox"][3] for c in self.cell_files)
        self.gds_w, self.gds_h = self.all_max_x - self.all_min_x, self.all_max_y - self.all_min_y

        self.native_dims = {}
        for entry in self.cell_files:
            try:
                with Image.open(entry["filepath"]) as img_hdr:
                    self.native_dims[entry["filename"]] = img_hdr.size
            except Exception:
                self.native_dims[entry["filename"]] = (self.max_disp_w, self.max_disp_h)

        self.current_idx = 0
        self.annotations = self.load_existing_annotations()
        self.exclusions = self.load_exclusions_file()

        # Mouse-state initializations (prevent AttributeError on early mouse events)
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
            except Exception:
                pass
        return {}

    def save_annotations_to_file(self):
        with open(self.output_json_path, 'w') as f:
            json.dump(self.annotations, f, indent=4)

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
        with open("manual_exclusions.json", "w") as f:
            json.dump(sorted(list(self.exclusions)), f, indent=4)

    def load_cell_at_index(self, idx: int):
        self.current_idx = idx
        cell_entry = self.cell_files[idx]
        self.native_w, self.native_h = self.native_dims.get(cell_entry["filename"], (self.max_disp_w, self.max_disp_h))

        preview_path = self.out_dir / "previews" / f"{self.wafer_id}_cell_{cell_entry['cell_data']['row']}-{cell_entry['cell_data']['col']}_preview.jpg"
        self.img_orig = cv2.imread(str(preview_path if preview_path.exists() else cell_entry["filepath"]))
        if self.img_orig is None:
            return

        self.orig_h, self.orig_w = self.img_orig.shape[:2]
        self.scale = min(self.max_disp_w / float(self.orig_w), self.max_disp_h / float(self.orig_h))
        self.display_width = int(self.orig_w * self.scale)
        self.display_height = int(self.orig_h * self.scale)
        self.img_disp = cv2.resize(self.img_orig, (self.display_width, self.display_height))

        if cell_entry["filename"] not in self.annotations:
            self.annotations[cell_entry["filename"]] = []
            self.save_annotations_to_file()

        # --- Diagnostic: verify that crop_pixel_to_gds round-trips the 4 GDS corners ---
        print(f"\n[DefectMapper] Loading cell {cell_entry['cell_data']['row']}-{cell_entry['cell_data']['col']} "
              f"(native {self.native_w}x{self.native_h}, preview {self.orig_w}x{self.orig_h})")
        self.transformer.verify_crop_corners(cell_entry["cell_data"], shave=self.shave, pad=self.pad)

        self.redraw_canvas()

    def redraw_canvas(self):
        panel_width = 320
        self.canvas = np.zeros((self.display_height, self.display_width + panel_width, 3), dtype=np.uint8)
        self.canvas[:, :self.display_width] = self.img_disp
        self.canvas[:, self.display_width:] = 40

        filename = self.cell_files[self.current_idx]["filename"]
        cv2.putText(self.canvas, "DEVICE METRIC STATUS", (self.display_width + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"Index: {self.current_idx + 1} / {len(self.cell_files)}", (self.display_width + 15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"File: {filename}", (self.display_width + 15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        legend_y = 120
        cv2.putText(self.canvas, "LEGEND GUIDE:", (self.display_width + 15, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        for idx, (label, l_color) in enumerate([
            ("Current Selected", (0, 165, 255)),
            ("Annotated / Active", (0, 120, 0)),
            ("Excluded / Damaged", (0, 0, 150)),
            ("Unvisited / Pending", (80, 80, 80))
        ]):
            ly = legend_y + 20 + idx * 20
            cv2.rectangle(self.canvas, (self.display_width + 15, ly - 10), (self.display_width + 27, ly + 2), l_color, -1)
            cv2.rectangle(self.canvas, (self.display_width + 15, ly - 10), (self.display_width + 27, ly + 2), (255, 255, 255), 1)
            cv2.putText(self.canvas, label, (self.display_width + 37, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        map_size, map_padding = 260, 15
        map_draw_size = map_size - 2 * map_padding
        map_x_start = self.display_width + 30
        map_y_start = max(250, self.display_height - map_size - 5)
        cv2.rectangle(self.canvas, (map_x_start - 10, map_y_start - 10), (map_x_start + map_size - 10, map_y_start + map_size - 10), (30, 30, 30), -1)
        cv2.rectangle(self.canvas, (map_x_start - 10, map_y_start - 10), (map_x_start + map_size - 10, map_y_start + map_size - 10), (100, 100, 100), 1)

        self.cell_map_bounds = []
        for idx, entry in enumerate(self.cell_files):
            min_x, min_y, max_x, max_y = entry["cell_data"]["bbox"]
            norm_x1 = (min_x - self.all_min_x) / (self.gds_w + 1e-9)
            norm_y1 = (min_y - self.all_min_y) / (self.gds_h + 1e-9)
            norm_x2 = (max_x - self.all_min_x) / (self.gds_w + 1e-9)
            norm_y2 = (max_y - self.all_min_y) / (self.gds_h + 1e-9)
            mx1 = int(map_x_start + min(norm_x1, norm_x2) * map_draw_size)
            my1 = int(map_y_start + (1.0 - max(norm_y1, norm_y2)) * map_draw_size)
            mx2 = int(map_x_start + max(norm_x1, norm_x2) * map_draw_size)
            my2 = int(map_y_start + (1.0 - min(norm_y1, norm_y2)) * map_draw_size)
            self.cell_map_bounds.append((mx1, my1, mx2, my2))

            is_current = (idx == self.current_idx)
            is_excluded = (entry["filename"] in self.exclusions)
            has_ann = (len(self.annotations.get(entry["filename"], [])) > 0)
            color = (0, 165, 255) if is_current else ((0, 0, 150) if is_excluded else ((0, 120, 0) if has_ann else (80, 80, 80)))
            thickness = -1 if (is_current or is_excluded or has_ann) else 1
            cv2.rectangle(self.canvas, (mx1, my1), (mx2, my2), color, thickness)
            if thickness == -1:
                cv2.rectangle(self.canvas, (mx1, my1), (mx2, my2), (200, 200, 200) if is_current else (40, 40, 40), 1)

        if filename in self.exclusions:
            cv2.rectangle(self.canvas, (0, 0), (self.display_width, self.display_height), (0, 0, 255), 4)

        for box in self.annotations.get(filename, []):
            x_tl, y_tl, w, h = box["box_px"]
            scale_down_x = self.display_width / float(self.native_w)
            scale_down_y = self.display_height / float(self.native_h)
            sc_x1, sc_y1 = int(x_tl * scale_down_x), int(y_tl * scale_down_y)
            sc_x2, sc_y2 = int((x_tl + w) * scale_down_x), int((y_tl + h) * scale_down_y)
            color = CLASS_COLORS.get(box["type"], (255, 255, 255))
            cv2.rectangle(self.canvas, (sc_x1, sc_y1), (sc_x2, sc_y2), color, 3)
            cv2.putText(
                self.canvas,
                f"{box['type']} (GDS X:{box.get('center_x_um', 0.0):.1f}, Y:{box.get('center_y_um', 0.0):.1f})",
                (sc_x1, sc_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )

        cv2.imshow("Device Defect Register", self.canvas)

    def handle_mouse(self, event, x, y, flags, param):
        if self.is_waiting_for_key:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= self.display_width:
                for idx, bounds in enumerate(self.cell_map_bounds):
                    bx1, by1, bx2, by2 = bounds
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        self.save_annotations_to_file()
                        self.load_cell_at_index(idx)
                        return
            else:
                self.drawing = True
                self.start_pt, self.current_pt = (x, y), (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            clamped_x = max(0, min(x, self.display_width - 1))
            clamped_y = max(0, min(y, self.display_height - 1))
            self.current_pt = (clamped_x, clamped_y)
            temp = self.canvas.copy()
            cv2.rectangle(temp, self.start_pt, self.current_pt, (0, 255, 255), 2)
            cv2.imshow("Device Defect Register", temp)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            w_disp = abs(self.current_pt[0] - self.start_pt[0])
            h_disp = abs(self.current_pt[1] - self.start_pt[1])
            if w_disp < 4 or h_disp < 4:
                self.redraw_canvas()
                return

            temp = self.canvas.copy()
            cv2.rectangle(temp, self.start_pt, self.current_pt, (0, 165, 255), 3)
            cv2.rectangle(temp, (self.display_width + 5, 105), (self.display_width + 315, 335), (0, 0, 180), -1)
            cv2.rectangle(temp, (self.display_width + 5, 105), (self.display_width + 315, 335), (0, 255, 255), 2)
            cv2.putText(temp, "CHOOSE DEFECT TYPE!", (self.display_width + 15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            classes = ["blister", "tear", "delamination", "particulate", "hole"]
            for idx, c_name in enumerate(classes):
                cy = 160 + idx * 22
                cv2.rectangle(temp, (self.display_width + 200, cy - 10), (self.display_width + 215, cy + 2), CLASS_COLORS[c_name], -1)
                cv2.rectangle(temp, (self.display_width + 200, cy - 10), (self.display_width + 215, cy + 2), (255, 255, 255), 1)
                cv2.putText(temp, f"[{idx+1}]: {c_name}", (self.display_width + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Device Defect Register", temp)

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
                # Scale display coordinates up to native crop pixel space
                scale_up_x = self.native_w / float(self.display_width)
                scale_up_y = self.native_h / float(self.display_height)

                orig_x1 = max(0, min(int(round(min(self.start_pt[0], self.current_pt[0]) * scale_up_x)), self.native_w))
                orig_y1 = max(0, min(int(round(min(self.start_pt[1], self.current_pt[1]) * scale_up_y)), self.native_h))
                orig_x2 = max(0, min(int(round(max(self.start_pt[0], self.current_pt[0]) * scale_up_x)), self.native_w))
                orig_y2 = max(0, min(int(round(max(self.start_pt[1], self.current_pt[1]) * scale_up_y)), self.native_h))

                filename = self.cell_files[self.current_idx]["filename"]
                cell_data = self.cell_files[self.current_idx]["cell_data"]

                # Map bounding box center — pass shave and pad explicitly to match extraction
                px_cx = orig_x1 + (orig_x2 - orig_x1) / 2.0
                px_cy = orig_y1 + (orig_y2 - orig_y1) / 2.0
                center_x_um, center_y_um = self.transformer.crop_pixel_to_gds(
                    px_cx, px_cy, cell_data, shave=self.shave, pad=self.pad
                )

                # --- FIX: map ALL FOUR pixel-space corners to GDS, not just TL/BR ---
                # Because the wafer (and the native crop frame) is rotated relative to
                # the GDS axes by self.transformer.alpha, an axis-aligned pixel box maps
                # to a ROTATED rectangle (parallelogram) in GDS microns. Reducing that to
                # a single width/height pair and reconstructing an axis-aligned GDS box
                # (as subtract_defects.py used to do) throws away the rotation and is the
                # source of the mask/stitched-image misalignment. We now store the full
                # rotated quadrilateral so the mask cut can reproduce it exactly.
                corners_px = [
                    (orig_x1, orig_y1),  # top-left
                    (orig_x2, orig_y1),  # top-right
                    (orig_x2, orig_y2),  # bottom-right
                    (orig_x1, orig_y2),  # bottom-left
                ]
                corners_gds = [
                    self.transformer.crop_pixel_to_gds(px, py, cell_data, shave=self.shave, pad=self.pad)
                    for px, py in corners_px
                ]
                corners_gds = [[round(float(gx), 3), round(float(gy), 3)] for gx, gy in corners_gds]

                # Axis-aligned width/height retained for display/sorting/back-compat only.
                # These DO NOT represent a valid axis-aligned GDS box when alpha != 0 —
                # subtract_defects.py must use corners_gds for the actual mask geometry.
                xs = [c[0] for c in corners_gds]
                ys = [c[1] for c in corners_gds]
                width_um = max(xs) - min(xs)
                height_um = max(ys) - min(ys)

                self.annotations[filename].append({
                    "type": assigned_class,
                    "box_px": [orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1],
                    "center_x_um": round(center_x_um, 3),
                    "center_y_um": round(center_y_um, 3),
                    "width_um": round(width_um, 3),
                    "height_um": round(height_um, 3),
                    "corners_gds": corners_gds
                })
                self.save_annotations_to_file()

            self.redraw_canvas()

    def stitch_and_save_wafer_layout(self):
        print("\nCompiling GDS-aligned physical wafer overview composite...")
        out_size = self.config.get("output_image_size", 4000)
        composite_canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        half = out_size / 2.0
        scale = (0.925 * half) / self.gds_R
        cv2.circle(composite_canvas, (int(half), int(half)), int(self.gds_R * scale), (60, 60, 60), 2, lineType=cv2.LINE_AA)

        for cell_entry in self.cell_files:
            cell_img = cv2.imread(str(cell_entry["filepath"]))
            if cell_img is None:
                continue

            for box in self.annotations.get(cell_entry["filename"], []):
                x, y, w, h = box["box_px"]
                color = CLASS_COLORS.get(box["type"], (255, 255, 255))
                p_x1, p_y1 = max(0, x - 40), max(0, y - 40)
                p_x2, p_y2 = min(cell_img.shape[1], x + w + 40), min(cell_img.shape[0], y + h + 40)
                cv2.rectangle(cell_img, (p_x1, p_y1), (p_x2, p_y2), color, 16)
                cv2.putText(cell_img, box["type"].upper(), (p_x1, p_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5, cv2.LINE_AA)

            min_x, min_y, max_x, max_y = cell_entry["cell_data"]["bbox"]
            pts_img = [self.transformer.transform_gds_to_target_img(gx, gy, out_size)
                       for gx, gy in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]]
            pts_img = np.array(pts_img)
            tx_min, ty_min = np.min(pts_img, axis=0)
            tx_max, ty_max = np.max(pts_img, axis=0)
            pt_x1, pt_y1 = int(round(tx_min)), int(round(ty_min))
            pt_x2, pt_y2 = int(round(tx_max)), int(round(ty_max))
            cell_w, cell_h = pt_x2 - pt_x1, pt_y2 - pt_y1
            if cell_w > 0 and cell_h > 0:
                composite_canvas[pt_y1:pt_y2, pt_x1:pt_x2] = cv2.resize(
                    cell_img, (cell_w, cell_h), interpolation=cv2.INTER_AREA
                )

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
            elif key in LEFT_ARROW_CODES or key in [ord('p'), ord('P')]:
                self.save_annotations_to_file()
                if self.current_idx > 0:
                    self.load_cell_at_index(self.current_idx - 1)
            elif key in [ord('x'), ord('X')]:
                filename = self.cell_files[self.current_idx]["filename"]
                if filename in self.exclusions:
                    self.exclusions.remove(filename)
                else:
                    self.exclusions.add(filename)
                self.save_exclusions_file()
                self.redraw_canvas()
            elif key in [ord('c'), ord('C')]:
                self.annotations[self.cell_files[self.current_idx]["filename"]] = []
                self.redraw_canvas()
                self.save_annotations_to_file()
            elif key == 27 or key in [ord('q'), ord('Q')]:
                self.save_annotations_to_file()
                self.save_exclusions_file()
                self.stitch_and_save_wafer_layout()
                break

        cv2.destroyAllWindows()