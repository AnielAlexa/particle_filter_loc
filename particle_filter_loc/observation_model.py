"""Wraps coarse (BoQ TRT) and fine (MatchAnything TRT) matchers."""

import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .geo_utils import ENUFrame, haversine_m


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == h:
        return frame
    s = min(w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return frame[y0:y0+s, x0:x0+s]


@dataclass
class CoarseResult:
    top_k_names: List[str]
    top_k_sims: List[float]
    top_k_indices: List[int]


@dataclass
class FineResult:
    lat: float
    lon: float
    inliers: int
    method: str  # "pnp" or "homography"
    heading_deg: Optional[float]
    patch_name: str
    mkpts_drone: Optional[np.ndarray] = None   # [M,2] keypoints in drone frame (320px)
    mkpts_patch: Optional[np.ndarray] = None   # [M,2] keypoints in patch pixel space
    H: Optional[np.ndarray] = None             # homography: drone(320) → patch(320) matcher space


def _import_match_module(script_dir: str):
    import sys
    sdir = str(script_dir)
    if sdir not in sys.path:
        sys.path.insert(0, sdir)
    src = Path(script_dir) / "3_match_video_dinov3.py"
    spec = importlib.util.spec_from_file_location("match_video_dinov3", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_patch_name(name: str) -> Tuple[int, int]:
    m = re.match(r"patch_(\d+)_(\d+)$", name)
    if not m:
        raise ValueError(f"Cannot parse patch name: {name!r}")
    return int(m.group(1)), int(m.group(2))


class ObservationModel:

    def __init__(self, config: dict, enu_frame: ENUFrame):
        self.enu = enu_frame
        self.cfg = config

        script_dir = Path(config["script_dir"])
        self.patches_dir = script_dir / config["patches_dir"]
        self.matcher_resolution = config.get("matcher_resolution", 320)
        self.camera_fx = config.get("camera_fx", 190.0)
        self.camera_fy = config.get("camera_fy", 190.0)
        self.camera_cx = config.get("camera_cx", self.matcher_resolution / 2.0)
        self.camera_cy = config.get("camera_cy", self.matcher_resolution / 2.0)
        self.altitude_m: float = 50.0   # updated continuously by ros2_pf_node each frame
        self.context_enabled = config.get("context_enabled", True)
        self.context_fraction = config.get("context_fraction", 0.4)
        self.fine_conf_threshold = config.get("fine_conf_threshold", 0.20)
        self.min_inliers_ransac = config.get("min_inliers_ransac", 4)
        self.min_inliers = config.get("min_inliers", 8)
        self.image_size = config.get("image_size", 256)
        self.grayscale = config.get("grayscale", True)

        # Load database
        db_path = script_dir / config["boq_database_path"]
        print(f"Loading BoQ descriptors from {db_path} ...")
        data = torch.load(str(db_path), map_location="cuda")
        if isinstance(data, dict):
            self.db_descriptors = data.get("descriptors", data.get("boq_descriptors", list(data.values())[0]))
        else:
            self.db_descriptors = data
        self.db_descriptors = F.normalize(self.db_descriptors.float().cuda(), dim=1)
        self.n_patches = self.db_descriptors.shape[0]

        # Patch names
        names_path = script_dir / config["patch_names_path"]
        with open(str(names_path)) as f:
            self.patch_names = [l.strip() for l in f if l.strip()]
        assert len(self.patch_names) == self.n_patches

        # GPS metadata
        meta_path = script_dir / config["gps_metadata_path"]
        with open(str(meta_path)) as f:
            self.gps_metadata: Dict = json.load(f)

        # Apply geolocation offset if provided (corrects satellite imagery bias vs RTK)
        lat_off = config.get("gps_offset_lat", 0.0)
        lon_off = config.get("gps_offset_lon", 0.0)
        if lat_off != 0.0 or lon_off != 0.0:
            for entry in self.gps_metadata.values():
                entry["lat"] += lat_off
                entry["lon"] += lon_off
                b = entry["bounds"]
                b["min_lat"] += lat_off;  b["max_lat"] += lat_off
                b["min_lon"] += lon_off;  b["max_lon"] += lon_off
            print(f"GPS offset applied: +{lat_off*111320:.1f}m N, +{lon_off*111320*0.707:.1f}m E")

        # Pre-compute patch centers in ENU
        self._patch_enu: List[Tuple[float, float]] = []
        for name in self.patch_names:
            meta = self.gps_metadata.get(name)
            if meta:
                e, n = self.enu.wgs84_to_enu(meta["lat"], meta["lon"])
                self._patch_enu.append((e, n))
            else:
                self._patch_enu.append((0.0, 0.0))
        self._patch_enu_np = np.array(self._patch_enu, dtype=np.float64)  # [N_patches, 2]

        # Load TRT engines
        mod = _import_match_module(config["script_dir"])

        boq_path = script_dir / config["boq_engine_path"]
        print(f"Loading coarse BoQ engine: {boq_path}")
        self.extractor = mod.DINOv3LoRABoQExtractorTRT(
            engine_path=str(boq_path),
            image_size=self.image_size,
            grayscale=self.grayscale,
        )

        fine_path = script_dir / config["fine_engine_path"]
        print(f"Loading fine matcher engine: {fine_path}")
        self.matcher = mod.TRTWrapper(str(fine_path))
        print("Models loaded.")

    # ------------------------------------------------------------------
    # Coarse matching
    # ------------------------------------------------------------------

    def coarse_match(self, frame_bgr: np.ndarray,
                     candidate_indices: Optional[List[int]] = None,
                     top_k: int = 5) -> CoarseResult:
        tensor = self.extractor.preprocess(frame_bgr)
        desc = self.extractor(tensor)
        desc = F.normalize(desc.float(), dim=1)

        if candidate_indices is not None and len(candidate_indices) > 0:
            db_sub = self.db_descriptors[candidate_indices]
            sims = (desc @ db_sub.T).squeeze(0)
            k_actual = min(top_k, len(candidate_indices))
            topk = sims.topk(k_actual)
            top_local = topk.indices.tolist()
            top_sims = topk.values.tolist()
            top_indices = [candidate_indices[i] for i in top_local]
        else:
            sims = (desc @ self.db_descriptors.T).squeeze(0)
            k_actual = min(top_k, self.n_patches)
            topk = sims.topk(k_actual)
            top_indices = topk.indices.tolist()
            top_sims = topk.values.tolist()

        top_names = [self.patch_names[i] for i in top_indices]
        return CoarseResult(top_k_names=top_names, top_k_sims=top_sims, top_k_indices=top_indices)

    # ------------------------------------------------------------------
    # Fine matching
    # ------------------------------------------------------------------

    def fine_match(self, frame_bgr: np.ndarray, patch_name: str,
                   context_fraction: float = 0.4) -> Optional[FineResult]:
        res = self.matcher_resolution

        # Build patch image + metadata
        if self.context_enabled:
            result = self._build_context_patch(patch_name, context_fraction)
            if result is None:
                return None
            patch_bgr, flat_meta = result
        else:
            raw = self.gps_metadata.get(patch_name)
            if raw is None:
                return None
            patch_path = self.patches_dir / (patch_name + ".png")
            patch_bgr = cv2.imread(str(patch_path))
            if patch_bgr is None:
                return None
            ph, pw = patch_bgr.shape[:2]
            b = raw["bounds"]
            flat_meta = {
                "min_lat": b["min_lat"], "max_lat": b["max_lat"],
                "min_lon": b["min_lon"], "max_lon": b["max_lon"],
                "patch_h": ph, "patch_w": pw,
            }

        patch_h = flat_meta["patch_h"]
        patch_w = flat_meta["patch_w"]

        # Center-crop drone frame to square, then resize
        frame_cropped = _center_crop_square(frame_bgr)
        q_gray = cv2.cvtColor(
            cv2.resize(frame_cropped, (res, res)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32) / 255.0
        p_gray = cv2.cvtColor(
            cv2.resize(patch_bgr, (res, res)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32) / 255.0
        q_np = q_gray[np.newaxis, np.newaxis]
        p_np = p_gray[np.newaxis, np.newaxis]

        try:
            out = self.matcher.infer(q_np, p_np)
        except Exception:
            return None

        mkpts0 = out.get("keypoints0")
        mkpts1 = out.get("keypoints1")
        mconf = out.get("mconf")

        if mkpts0 is None or mkpts1 is None or mconf is None or len(mconf) == 0:
            return None

        mask = mconf > self.fine_conf_threshold
        if mask.sum() < self.min_inliers_ransac:
            return None

        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]

        # Scale patch keypoints to composite pixel space
        mkpts1_patch = mkpts1 * np.array([[patch_w / res, patch_h / res]], dtype=np.float32)

        # Homography for visualization (drone 320px → patch 320px matcher space)
        vis_H = None
        if len(mkpts0) >= 4:
            vis_H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)

        # --- PnP ---
        lat, lon, inliers, method, heading_deg = None, None, 0, "homography", None

        if len(mkpts0) >= self.min_inliers_ransac:
            pnp_result = self._pnp_refine_gps(mkpts0, mkpts1_patch, flat_meta)
            if pnp_result is not None:
                lat, lon, heading_deg, inliers = pnp_result
                method = "pnp"

        # --- Homography fallback ---
        if lat is None:
            if len(mkpts0) < 4:
                return None
            H, hm = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
            if H is None:
                return None
            inliers = int(hm.sum())
            if inliers < self.min_inliers_ransac:
                return None
            h_result = self._homography_center_gps(H, flat_meta, res)
            if h_result is None:
                return None
            lat, lon = h_result
            method = "homography"

        if inliers < self.min_inliers:
            return None

        return FineResult(
            lat=lat, lon=lon, inliers=inliers, method=method,
            heading_deg=heading_deg, patch_name=patch_name,
            mkpts_drone=mkpts0, mkpts_patch=mkpts1_patch,
            H=vis_H,
        )

    # ------------------------------------------------------------------
    # PnP refinement with heading extraction
    # ------------------------------------------------------------------

    def _pnp_refine_gps(
        self,
        mkpts_drone: np.ndarray,
        mkpts_patch_px: np.ndarray,
        meta: dict,
    ) -> Optional[Tuple[float, float, float]]:
        """Returns (lat, lon, heading_deg) or None."""
        min_lat = meta["min_lat"]
        max_lat = meta["max_lat"]
        min_lon = meta["min_lon"]
        max_lon = meta["max_lon"]
        patch_h = meta["patch_h"]
        patch_w = meta["patch_w"]

        lats = max_lat - mkpts_patch_px[:, 1] * (max_lat - min_lat) / (patch_h - 1)
        lons = min_lon + mkpts_patch_px[:, 0] * (max_lon - min_lon) / (patch_w - 1)

        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0

        def gps_to_enu(lat, lon):
            dlat = lat - center_lat
            dlon = lon - center_lon
            x = dlon * math.cos(math.radians(center_lat)) * 111319.5
            y = dlat * 111319.5
            return x, y

        obj_pts = []
        for lat, lon in zip(lats, lons):
            x, y = gps_to_enu(lat, lon)
            obj_pts.append([x, y, 0.0])
        obj_pts = np.array(obj_pts, dtype=np.float32)

        img_pts = mkpts_drone.astype(np.float32)
        K = np.array([[self.camera_fx, 0,             self.camera_cx],
                      [0,             self.camera_fy, self.camera_cy],
                      [0,             0,              1             ]], dtype=np.float32)

        tvec_init = np.array([[0.0], [0.0], [self.altitude_m]], dtype=np.float32)
        try:
            ok, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
                obj_pts, img_pts, K, None,
                None, tvec_init, True,
                100, 8.0, 0.99, None,
                cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            return None

        if not ok or inlier_idx is None or len(inlier_idx) < self.min_inliers_ransac:
            return None

        # Refine pose using inliers only
        inlier_obj = obj_pts[inlier_idx.flatten()]
        inlier_img = img_pts[inlier_idx.flatten()]
        try:
            ok2, rvec, tvec = cv2.solvePnP(
                inlier_obj, inlier_img, K, None, rvec, tvec, True,
                cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            ok2 = False
        if not ok2:
            return None

        R, _ = cv2.Rodrigues(rvec)
        cam_pos = (-R.T @ tvec).flatten()

        # Sanity check: PnP altitude must be within 50% of live altimeter reading
        if self.altitude_m > 0.0 and abs(cam_pos[2] - self.altitude_m) > self.altitude_m * 0.5:
            return None

        cam_lat = center_lat + cam_pos[1] / 111319.5
        cam_lon = center_lon + cam_pos[0] / (111319.5 * math.cos(math.radians(center_lat)))

        # Extract heading from rotation matrix (nadir camera assumption)
        heading_rad = math.atan2(R[1, 0], R[0, 0])
        heading_deg = math.degrees(heading_rad) % 360.0

        return cam_lat, cam_lon, heading_deg, len(inlier_idx)

    # ------------------------------------------------------------------
    # Homography fallback
    # ------------------------------------------------------------------

    def _homography_center_gps(
        self, H: np.ndarray, meta: dict, drone_res: int,
    ) -> Optional[Tuple[float, float]]:
        cx = drone_res / 2.0
        cy = drone_res / 2.0
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        pt_res = cv2.perspectiveTransform(pt, H)[0][0]

        min_lat = meta["min_lat"]
        max_lat = meta["max_lat"]
        min_lon = meta["min_lon"]
        max_lon = meta["max_lon"]
        patch_h = meta["patch_h"]
        patch_w = meta["patch_w"]

        px = float(pt_res[0]) * patch_w / drone_res
        py = float(pt_res[1]) * patch_h / drone_res
        px = max(0.0, min(float(patch_w - 1), px))
        py = max(0.0, min(float(patch_h - 1), py))

        lat = max_lat - py * (max_lat - min_lat) / (patch_h - 1)
        lon = min_lon + px * (max_lon - min_lon) / (patch_w - 1)
        return lat, lon

    # ------------------------------------------------------------------
    # Context patch stitching
    # ------------------------------------------------------------------

    def _build_context_patch(
        self, center_name: str, context_fraction: float,
    ) -> Optional[Tuple[np.ndarray, dict]]:
        raw = self.gps_metadata.get(center_name)
        if raw is None:
            return None

        try:
            row, col = _parse_patch_name(center_name)
        except ValueError:
            return None

        center_bounds = raw["bounds"]

        center_path = self.patches_dir / (center_name + ".png")
        center_img = cv2.imread(str(center_path))
        if center_img is None:
            return None
        tile_h, tile_w = center_img.shape[:2]

        tile_lat_span = center_bounds["max_lat"] - center_bounds["min_lat"]
        tile_lon_span = center_bounds["max_lon"] - center_bounds["min_lon"]

        lat_per_px = tile_lat_span / (tile_h - 1)
        lon_per_px = tile_lon_span / (tile_w - 1)

        ext_lat = context_fraction * tile_lat_span
        ext_lon = context_fraction * tile_lon_span
        comp_min_lat = center_bounds["min_lat"] - ext_lat
        comp_max_lat = center_bounds["max_lat"] + ext_lat
        comp_min_lon = center_bounds["min_lon"] - ext_lon
        comp_max_lon = center_bounds["max_lon"] + ext_lon

        comp_h = round((comp_max_lat - comp_min_lat) / lat_per_px) + 1
        comp_w = round((comp_max_lon - comp_min_lon) / lon_per_px) + 1

        composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)

        neighbours = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                is_center = (dr == 0 and dc == 0)
                nname = f"patch_{row + dr}_{col + dc}"
                if nname not in self.gps_metadata:
                    continue
                nb = self.gps_metadata[nname]["bounds"]
                if (nb["max_lat"] <= comp_min_lat or nb["min_lat"] >= comp_max_lat or
                        nb["max_lon"] <= comp_min_lon or nb["min_lon"] >= comp_max_lon):
                    continue
                neighbours.append((nname, nb, is_center))

        neighbours.sort(key=lambda x: x[2])

        for nname, nb, is_center in neighbours:
            if is_center:
                timg = center_img
            else:
                tpath = self.patches_dir / (nname + ".png")
                timg = cv2.imread(str(tpath))
                if timg is None:
                    continue

            th, tw = timg.shape[:2]
            x_off = round((nb["min_lon"] - comp_min_lon) / lon_per_px)
            y_off = round((comp_max_lat - nb["max_lat"]) / lat_per_px)

            src_x0 = max(0, -x_off);    dst_x0 = max(0, x_off)
            src_y0 = max(0, -y_off);    dst_y0 = max(0, y_off)
            src_x1 = min(tw, comp_w - x_off)
            src_y1 = min(th, comp_h - y_off)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            if src_x1 > src_x0 and src_y1 > src_y0:
                composite[dst_y0:dst_y1, dst_x0:dst_x1] = timg[src_y0:src_y1, src_x0:src_x1]

        flat_meta = {
            "min_lat": comp_min_lat,
            "max_lat": comp_max_lat,
            "min_lon": comp_min_lon,
            "max_lon": comp_max_lon,
            "patch_h": comp_h,
            "patch_w": comp_w,
        }
        return composite, flat_meta

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def get_patch_center_enu(self, name: str) -> Tuple[float, float]:
        idx = self.patch_names.index(name)
        return self._patch_enu[idx]

    def get_indices_within_radius(self, east: float, north: float, radius_m: float) -> List[int]:
        dx = self._patch_enu_np[:, 0] - east
        dy = self._patch_enu_np[:, 1] - north
        dist2 = dx ** 2 + dy ** 2
        return np.where(dist2 <= radius_m ** 2)[0].tolist()
