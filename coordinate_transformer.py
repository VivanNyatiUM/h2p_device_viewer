import numpy as np
import math
import cv2

def umeyama_rigid_registration(source_pts: np.ndarray, target_pts: np.ndarray):
    """
    Solves a 2D rigid-body similarity transformation (Kabsch-Umeyama).
    Maps source coordinates to target coordinates: Y = scale * (R @ X) + t
    
    Parameters:
    -----------
    source_pts : np.ndarray of shape (N, 2)
        Unrotated base physical coordinates.
    target_pts : np.ndarray of shape (N, 2)
        Nominal target GDS coordinates (microns).
        
    Returns:
    --------
    scale : float
        Isotropic scaling multiplier.
    R : np.ndarray of shape (2, 2)
        2D rotation matrix.
    t : np.ndarray of shape (2, 1)
        Translation offsets (microns).
    rmsd : float
        Root-mean-square deviation of point pairs.
    """
    X = source_pts.T  # Shape (2, N)
    Y = target_pts.T  # Shape (2, N)
    m, n = X.shape

    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)

    var_x = np.sum((X - mu_x) ** 2) / n
    if var_x < 1e-9:
        return 1.0, np.eye(2), np.zeros((2, 1)), 0.0

    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / n

    U, D, VH = np.linalg.svd(cov_xy)

    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1

    R = U @ S @ VH
    scale = np.trace(np.diag(D) @ S) / var_x
    t = mu_y - scale * R @ mu_x

    # Compute Root-Mean-Square Deviation
    Y_pred = scale * (R @ X) + t
    rmsd = np.sqrt(np.mean(np.sum((Y - Y_pred) ** 2, axis=0)))

    return float(scale), R, t, float(rmsd)


class WaferTransformer:
    """Handles isomorphic bidirectional coordinate mappings: canvas px <=> GDS um."""
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

    def canvas_to_gds(self, x_can: float, y_can: float):
        dx = x_can - self.xc
        dy = self.yc - y_can

        dx_rot = dx * self._cos_a - dy * self._sin_a
        dy_rot = dx * self._sin_a + dy * self._cos_a

        gy_eff = dy_rot * self.S_y + self.y_offset
        gx_eff = dx_rot * self.S_x + self.shear * (dy_rot * self.S_y) + self.x_offset

        x_gds = gx_eff + self.xg_c
        y_gds = gy_eff + self.yg_c
        return x_gds, y_gds

    def gds_to_canvas(self, x_gds: float, y_gds: float):
        gx_centered = x_gds - self.xg_c
        gy_centered = y_gds - self.yg_c

        dy_rot = (gy_centered - self.y_offset) / self.S_y
        dx_rot = ((gx_centered - self.x_offset) - self.shear * (gy_centered - self.y_offset)) / self.S_x

        dx = dx_rot * self._cos_a + dy_rot * self._sin_a
        dy = -dx_rot * self._sin_a + dy_rot * self._cos_a

        x_can = self.xc + dx
        y_can = self.yc - dy
        return x_can, y_can

    def transform_gds_to_target_img(self, x_gds: float, y_gds: float, out_size: int):
        half  = out_size / 2.0
        scale = (0.925 * half) / self.Rg
        gx_centered = x_gds - self.xg_c
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

    def crop_pixel_to_gds(self, px_x: float, px_y: float, cell_data: dict, shave: int = 10, pad: int = 200) -> tuple[float, float]:
        """
        Maps a pixel coordinate inside a cropped cell image back to absolute GDS microns.
        Bypasses cell bounding box inflation by using the aligned wafer transformation directly.

        IMPORTANT: shave and pad must match the values used during Stage 1 extraction exactly.
        Pass them explicitly from the DeviceDefectMapperTool to avoid default mismatch.
        """
        min_x, min_y, max_x, max_y = cell_data["bbox"]

        # 1. Map GDS boundaries to canvas pixels to find rotation/crop offsets
        pts_canvas = np.array([self.gds_to_canvas(gx, gy) for gx, gy in [
            (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
        ]])
        cx_min, cy_min = np.min(pts_canvas, axis=0)
        cx_max, cy_max = np.max(pts_canvas, axis=0)

        # Reconstruct the exact padded crop bounding box used during Stage 1 slicing
        x1 = int(np.floor(cx_min)) - pad
        y1 = int(np.floor(cy_min)) - pad
        x2 = int(np.ceil(cx_max))  + pad
        y2 = int(np.ceil(cy_max))  + pad

        # Dimensions of the padded local canvas (matches local_w, local_h in extraction)
        local_w = x2 - x1
        local_h = y2 - y1

        # Reconstruct the exact rotation center used during Stage 1 slicing
        cell_center_canvas = np.mean(pts_canvas, axis=0)
        crop_center_x = cell_center_canvas[0] - x1
        crop_center_y = cell_center_canvas[1] - y1

        # Build the same rotation matrix M as used in extraction
        M = cv2.getRotationMatrix2D(
            (crop_center_x, crop_center_y),
            self.alpha * 180.0 / np.pi,
            1.0
        )

        # Rotate the 4 GDS corner projections into the rotated-local frame
        pts_local = np.column_stack([
            pts_canvas[:, 0] - x1,
            pts_canvas[:, 1] - y1,
            np.ones(4)
        ])
        pts_rotated_local = (M @ pts_local.T).T
        rx_min, ry_min = np.min(pts_rotated_local, axis=0)

        # 2. Reconstruct crop_x1 / crop_y1 with IDENTICAL clamping to the extraction pass
        #    (max(0, min(int(round(...)) + shave, local_dim - 1)))
        crop_x1 = max(0, min(int(round(rx_min)) + shave, local_w - 1))
        crop_y1 = max(0, min(int(round(ry_min)) + shave, local_h - 1))

        # 3. Map the native-crop pixel coordinate back to the rotated-local canvas frame
        rx_local = px_x + crop_x1
        ry_local = px_y + crop_y1

        # 4. Un-rotate from rotated-local frame back to padded-local frame
        #    cv2.getRotationMatrix2D with angle θ (in image/Y-down space) applies:
        #      [cos θ,  sin θ, tx]
        #      [-sin θ, cos θ, ty]
        #    The inverse (un-rotation by θ) in the same Y-down convention is:
        #      [cos θ, -sin θ]
        #      [sin θ,  cos θ]
        #    which is standard counter-clockwise rotation by θ in math coords — matching
        #    math.cos(-α)/sin(-α) only for α=0. Use the transpose of M's 2x2 block instead
        #    to guarantee exact inversion of whatever M computed.
        M2x2 = M[:2, :2]           # [[cos α,  sin α], [-sin α, cos α]]  (image Y-down)
        M2x2_inv = M2x2.T          # [[cos α, -sin α], [ sin α, cos α]]  exact inverse

        dx = rx_local - crop_center_x
        dy = ry_local - crop_center_y

        ux = crop_center_x + M2x2_inv[0, 0] * dx + M2x2_inv[0, 1] * dy
        uy = crop_center_y + M2x2_inv[1, 0] * dx + M2x2_inv[1, 1] * dy

        # 5. Map back to global canvas pixel coordinates
        x_can = ux + x1
        y_can = uy + y1

        # 6. Convert global canvas pixel to GDS microns
        return self.canvas_to_gds(x_can, y_can)

    def verify_crop_corners(self, cell_data: dict, shave: int = 10, pad: int = 200) -> float:
        """
        Diagnostic: projects the 4 GDS bbox corners through the forward extraction
        geometry (gds_to_canvas → rotate → crop) to find their expected native-crop
        pixel coordinates, then maps them back through crop_pixel_to_gds and reports
        the round-trip residual in microns.

        Returns the max corner residual (µm). Prints a per-corner breakdown.
        """
        min_x, min_y, max_x, max_y = cell_data["bbox"]
        gds_corners = [
            (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
        ]
        corner_labels = ["BL", "BR", "TR", "TL"]

        pts_canvas = np.array([self.gds_to_canvas(gx, gy) for gx, gy in gds_corners])
        cx_min, cy_min = np.min(pts_canvas, axis=0)
        cx_max, cy_max = np.max(pts_canvas, axis=0)

        x1 = int(np.floor(cx_min)) - pad
        y1 = int(np.floor(cy_min)) - pad
        x2 = int(np.ceil(cx_max))  + pad
        y2 = int(np.ceil(cy_max))  + pad
        local_w = x2 - x1
        local_h = y2 - y1

        cell_center_canvas = np.mean(pts_canvas, axis=0)
        crop_center_x = cell_center_canvas[0] - x1
        crop_center_y = cell_center_canvas[1] - y1

        M = cv2.getRotationMatrix2D(
            (crop_center_x, crop_center_y),
            self.alpha * 180.0 / np.pi,
            1.0
        )

        pts_local = np.column_stack([
            pts_canvas[:, 0] - x1,
            pts_canvas[:, 1] - y1,
            np.ones(4)
        ])
        pts_rotated_local = (M @ pts_local.T).T
        rx_min, ry_min = np.min(pts_rotated_local, axis=0)

        crop_x1 = max(0, min(int(round(rx_min)) + shave, local_w - 1))
        crop_y1 = max(0, min(int(round(ry_min)) + shave, local_h - 1))

        max_residual = 0.0
        print(f"  [CornerVerify] Cell bbox=({min_x:.0f},{min_y:.0f},{max_x:.0f},{max_y:.0f})  shave={shave} pad={pad}")
        for label, (gx_expected, gy_expected), pt_rot in zip(corner_labels, gds_corners, pts_rotated_local):
            # Native-crop pixel coordinate this corner maps to
            px_x = pt_rot[0] - crop_x1
            px_y = pt_rot[1] - crop_y1

            gx_rt, gy_rt = self.crop_pixel_to_gds(px_x, px_y, cell_data, shave=shave, pad=pad)
            err = math.hypot(gx_rt - gx_expected, gy_rt - gy_expected)
            max_residual = max(max_residual, err)
            print(f"    [{label}] expected=({gx_expected:.2f},{gy_expected:.2f})  got=({gx_rt:.2f},{gy_rt:.2f})  residual={err:.4f} µm")

        print(f"  [CornerVerify] Max corner residual: {max_residual:.4f} µm")
        return max_residual

    def run_self_test(self):
        test_gds_points = [(0.0, 0.0), (1500.0, -1500.0), (-self.Rg * 0.5, self.Rg * 0.5)]
        passed_all = True
        for gx, gy in test_gds_points:
            cx, cy = self.gds_to_canvas(gx, gy)
            gx_rt, gy_rt = self.canvas_to_gds(cx, cy)
            if abs(gx - gx_rt) > 1e-4 or abs(gy - gy_rt) > 1e-4:
                passed_all = False
        if passed_all:
            print("[Transformer Self-Test] Success. Coordinate mappings are isomorphic.")