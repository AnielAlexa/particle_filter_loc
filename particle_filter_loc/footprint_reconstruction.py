"""Satellite footprint reconstruction from GPS + heading + altitude."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .camera_footprint import compute_footprint_corners_gps
from .geo_utils import ENUFrame


@dataclass
class FootprintReconstruction:
    satellite_crop: np.ndarray          # BGR, warped to match drone FOV
    footprint_corners_gps: np.ndarray   # [4,2] (lat, lon)
    footprint_w_m: float
    footprint_h_m: float
    source_tiles: List[str] = field(default_factory=list)
    mosaic_bgr: Optional[np.ndarray] = None       # raw North-up stitched mosaic
    mosaic_meta: Optional[dict] = None             # {min_lat, max_lat, min_lon, max_lon, h, w}
    mosaic_rotated: Optional[np.ndarray] = None    # heading-aligned square crop, no black
    rotation_center_px: Optional[Tuple[float, float]] = None  # (cx, cy) in North-up mosaic
    heading_deg: float = 0.0                       # heading used for rotation
    warp_M_inv: Optional[np.ndarray] = None        # inverse perspective: satellite_crop px → mosaic px
    # Inverse affine: mosaic_rotated px → North-up mosaic px (2x3 matrix)
    rot_crop_M_inv: Optional[np.ndarray] = None


class SatelliteFootprintReconstructor:
    """Stitches satellite tiles and warps to match drone camera footprint."""

    def __init__(self, gps_metadata: Dict, patches_dir: Path, enu_frame: ENUFrame):
        self.gps_metadata = gps_metadata
        self.patches_dir = Path(patches_dir)
        self.enu = enu_frame

        # Cache tile images (LRU-style via dict)
        self._tile_cache: Dict[str, np.ndarray] = {}
        self._max_cache = 200

        # Precompute tile bounds arrays for fast overlap testing
        self._tile_names: List[str] = []
        self._tile_min_lat: List[float] = []
        self._tile_max_lat: List[float] = []
        self._tile_min_lon: List[float] = []
        self._tile_max_lon: List[float] = []

        for name, meta in gps_metadata.items():
            b = meta["bounds"]
            self._tile_names.append(name)
            self._tile_min_lat.append(b["min_lat"])
            self._tile_max_lat.append(b["max_lat"])
            self._tile_min_lon.append(b["min_lon"])
            self._tile_max_lon.append(b["max_lon"])

        self._tile_min_lat_arr = np.array(self._tile_min_lat)
        self._tile_max_lat_arr = np.array(self._tile_max_lat)
        self._tile_min_lon_arr = np.array(self._tile_min_lon)
        self._tile_max_lon_arr = np.array(self._tile_max_lon)

        # Compute GSD from first tile
        first_name = self._tile_names[0]
        first_b = gps_metadata[first_name]["bounds"]
        sample_img = self._load_tile(first_name)
        if sample_img is not None:
            self._tile_h, self._tile_w = sample_img.shape[:2]
        else:
            self._tile_h, self._tile_w = 400, 400
        self._lat_per_px = (first_b["max_lat"] - first_b["min_lat"]) / (self._tile_h - 1)
        self._lon_per_px = (first_b["max_lon"] - first_b["min_lon"]) / (self._tile_w - 1)

    def _load_tile(self, name: str) -> Optional[np.ndarray]:
        if name in self._tile_cache:
            return self._tile_cache[name]
        path = self.patches_dir / (name + ".png")
        img = cv2.imread(str(path))
        if img is None:
            return None
        if len(self._tile_cache) >= self._max_cache:
            # Evict oldest
            oldest = next(iter(self._tile_cache))
            del self._tile_cache[oldest]
        self._tile_cache[name] = img
        return img

    def reconstruct(
        self,
        lat: float, lon: float,
        altitude_m: float, heading_deg: float,
        fx: float, fy: float,
        img_w: int, img_h: int,
        output_size: Tuple[int, int] = (480, 640),
        mosaic_context_scale: float = 2.0,
    ) -> Optional[FootprintReconstruction]:
        """Reconstruct satellite view matching drone camera footprint.

        Args:
            output_size: (height, width) of output image (satellite_crop).
            mosaic_context_scale: how many times the footprint diagonal
                the heading-aligned mosaic square should cover.  Default 2.0
                means the square side = 2x the footprint diagonal.
        """
        footprint_w_m = altitude_m * img_w / fx
        footprint_h_m = altitude_m * img_h / fy

        # 1. Compute footprint corners in GPS (for satellite_crop perspective warp)
        corners_gps = compute_footprint_corners_gps(
            fx, fy, img_w, img_h, altitude_m, heading_deg,
            lat, lon, self.enu,
        )

        # 2. Determine mosaic extent.
        #    The heading-aligned output is a square of side S (meters).
        #    To guarantee no black after any rotation, the North-up mosaic
        #    must cover a circle of radius S*sqrt(2)/2 around the drone.
        footprint_diag_m = math.sqrt(footprint_w_m**2 + footprint_h_m**2)
        square_side_m = footprint_diag_m * mosaic_context_scale
        mosaic_radius_m = square_side_m * math.sqrt(2) / 2.0

        lat_rad = math.radians(lat)
        radius_lat = mosaic_radius_m / 111320.0
        radius_lon = mosaic_radius_m / (111320.0 * math.cos(lat_rad))

        min_lat = lat - radius_lat
        max_lat = lat + radius_lat
        min_lon = lon - radius_lon
        max_lon = lon + radius_lon

        # 3. Stitch mosaic covering the circle bounding box
        mosaic, mosaic_meta, source_tiles = self._stitch_mosaic(
            min_lat, max_lat, min_lon, max_lon,
        )
        if mosaic is None:
            return None

        m_min_lat = mosaic_meta["min_lat"]
        m_max_lat = mosaic_meta["max_lat"]
        m_min_lon = mosaic_meta["min_lon"]
        m_max_lon = mosaic_meta["max_lon"]
        m_h = mosaic_meta["h"]
        m_w = mosaic_meta["w"]

        # Drone position in North-up mosaic pixel space
        center_px_x = (lon - m_min_lon) / (m_max_lon - m_min_lon) * (m_w - 1)
        center_px_y = (m_max_lat - lat) / (m_max_lat - m_min_lat) * (m_h - 1)

        # 4. Perspective warp for satellite_crop (unchanged logic)
        src_pts = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            c_lat, c_lon = corners_gps[i]
            px = (c_lon - m_min_lon) / (m_max_lon - m_min_lon) * (m_w - 1)
            py = (m_max_lat - c_lat) / (m_max_lat - m_min_lat) * (m_h - 1)
            src_pts[i] = [px, py]

        out_h, out_w = output_size
        dst_pts = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        warped = cv2.warpPerspective(
            mosaic, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # 5. Heading-aligned square mosaic: rotate + crop in one warpAffine.
        #    Output is S×S pixels centered on drone, "up" = heading direction.
        #    The pre-rotation mosaic is large enough that no black appears.
        gsd_lon = (m_max_lon - m_min_lon) / (m_w - 1) * 111320.0 * math.cos(lat_rad)
        gsd_lat = (m_max_lat - m_min_lat) / (m_h - 1) * 111320.0
        gsd = (gsd_lon + gsd_lat) / 2.0  # meters per pixel
        S = max(64, int(round(square_side_m / gsd)))

        # Build combined affine: rotate by heading around drone center,
        # then translate so drone center lands at (S/2, S/2).
        rot_mat = cv2.getRotationMatrix2D(
            (float(center_px_x), float(center_px_y)), heading_deg, 1.0,
        )
        # After rotation the center stays at (center_px_x, center_px_y).
        # Shift so it lands at the middle of the S×S output.
        rot_mat[0, 2] += S / 2.0 - center_px_x
        rot_mat[1, 2] += S / 2.0 - center_px_y

        mosaic_rot = cv2.warpAffine(
            mosaic, rot_mat, (S, S),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Inverse affine: mosaic_rot px → North-up mosaic px
        M_fwd_3x3 = np.vstack([rot_mat, [0.0, 0.0, 1.0]])
        rot_crop_M_inv = np.linalg.inv(M_fwd_3x3)[:2]  # [2, 3]

        return FootprintReconstruction(
            satellite_crop=warped,
            footprint_corners_gps=corners_gps,
            footprint_w_m=footprint_w_m,
            footprint_h_m=footprint_h_m,
            source_tiles=source_tiles,
            mosaic_bgr=mosaic,
            mosaic_meta=mosaic_meta,
            mosaic_rotated=mosaic_rot,
            rotation_center_px=(float(center_px_x), float(center_px_y)),
            heading_deg=heading_deg,
            warp_M_inv=M_inv,
            rot_crop_M_inv=rot_crop_M_inv,
        )

    def _stitch_mosaic(
        self,
        min_lat: float, max_lat: float,
        min_lon: float, max_lon: float,
    ) -> Tuple[Optional[np.ndarray], dict, List[str]]:
        """Stitch tiles overlapping the given GPS bounding box into a mosaic."""
        # Find overlapping tiles (vectorized)
        overlap = (
            (self._tile_max_lat_arr > min_lat) &
            (self._tile_min_lat_arr < max_lat) &
            (self._tile_max_lon_arr > min_lon) &
            (self._tile_min_lon_arr < max_lon)
        )
        indices = np.where(overlap)[0]

        if len(indices) == 0:
            return None, {}, []

        # Compute mosaic dimensions
        comp_h = round((max_lat - min_lat) / self._lat_per_px) + 1
        comp_w = round((max_lon - min_lon) / self._lon_per_px) + 1
        comp_h = max(1, comp_h)
        comp_w = max(1, comp_w)

        composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
        source_tiles = []

        for idx in indices:
            name = self._tile_names[idx]
            timg = self._load_tile(name)
            if timg is None:
                continue

            source_tiles.append(name)
            nb_min_lat = self._tile_min_lat[idx]
            nb_max_lat = self._tile_max_lat[idx]
            nb_min_lon = self._tile_min_lon[idx]

            th, tw = timg.shape[:2]
            x_off = round((nb_min_lon - min_lon) / self._lon_per_px)
            y_off = round((max_lat - nb_max_lat) / self._lat_per_px)

            src_x0 = max(0, -x_off);    dst_x0 = max(0, x_off)
            src_y0 = max(0, -y_off);    dst_y0 = max(0, y_off)
            src_x1 = min(tw, comp_w - x_off)
            src_y1 = min(th, comp_h - y_off)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            if src_x1 > src_x0 and src_y1 > src_y0:
                composite[dst_y0:dst_y1, dst_x0:dst_x1] = timg[src_y0:src_y1, src_x0:src_x1]

        meta = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "h": comp_h,
            "w": comp_w,
        }
        return composite, meta, source_tiles
