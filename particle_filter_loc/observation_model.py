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
    # Flow geometry metrics (computed from matched keypoints)
    flow_consistency: float = 0.0   # circular resultant R of flow angles [0,1]
    flow_magnitude_cv: float = 1.0  # coefficient of variation of flow magnitudes
    inlier_ratio: float = 0.0       # inliers / total_confident_matches
    flow_heading_deg: Optional[float] = None  # dominant flow direction (degrees)
    n_total_matches: int = 0        # confidence-filtered matches before RANSAC


def compute_flow_metrics(
    mkpts_drone: np.ndarray, mkpts_patch: np.ndarray,
) -> dict:
    """Compute geometric quality metrics from matched keypoint pairs.

    Both arrays are [N, 2] in pixel coordinates (matcher resolution space).
    Computes on pre-RANSAC confidence-filtered matches for maximum signal.
    """
    if mkpts_drone is None or mkpts_patch is None or len(mkpts_drone) < 3:
        return {"flow_consistency": 0.0, "flow_magnitude_cv": 1.0,
                "flow_heading_deg": None}

    flow = mkpts_patch - mkpts_drone  # [N, 2]
    angles = np.arctan2(flow[:, 1], flow[:, 0])
    magnitudes = np.linalg.norm(flow, axis=1)

    # Flow consistency: circular resultant length R
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    R = float(np.sqrt(mean_sin**2 + mean_cos**2))  # [0, 1]

    # Flow magnitude coefficient of variation
    mag_mean = float(np.mean(magnitudes))
    mag_std = float(np.std(magnitudes))
    cv = mag_std / max(mag_mean, 1e-6)

    # Dominant flow direction
    flow_heading = float(np.degrees(np.arctan2(mean_sin, mean_cos))) % 360.0

    return {
        "flow_consistency": R,
        "flow_magnitude_cv": cv,
        "flow_heading_deg": flow_heading,
    }


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

        # ── Coarse matcher: VLAD TRT / VLAD PyTorch / BoQ TRT ──────────
        use_vlad = config.get("use_vlad", False)
        use_vlad_trt = config.get("use_vlad_trt", False)

        if use_vlad_trt:
            # TRT engine with VLAD+PCA baked in — outputs [1, 768] directly
            mod = _import_match_module(config["script_dir"])
            engine_path = script_dir / config["vlad_trt_engine_path"]
            print(f"Loading VLAD+PCA TRT engine: {engine_path}")
            self.extractor = mod.DINOv3VLADPCAExtractorTRT(
                engine_path=str(engine_path),
                image_size=self.image_size,
                grayscale=self.grayscale,
            )
            # Use the same VLAD database projected through PCA
            db_path = script_dir / config["vlad_database_path"]
            names_path = script_dir / config["vlad_patch_names_path"]
        elif use_vlad:
            from .vlad_extractor import DINOv3VLADExtractor
            codebook_path = script_dir / config["vlad_codebook_path"]
            self.extractor = DINOv3VLADExtractor(
                codebook_path=str(codebook_path),
                layer_idx=config.get("vlad_layer_idx", 10),
                image_size=self.image_size,
                grayscale=self.grayscale,
                device="cuda",
            )
            db_path = script_dir / config["vlad_database_path"]
            names_path = script_dir / config["vlad_patch_names_path"]
        else:
            db_path = script_dir / config["boq_database_path"]
            names_path = script_dir / config["patch_names_path"]

        # Load descriptor database
        coarse_label = "VLAD TRT" if use_vlad_trt else ("VLAD" if use_vlad else "BoQ")
        print(f"Loading {coarse_label} descriptors from {db_path} …")
        data = torch.load(str(db_path), map_location="cuda")
        if isinstance(data, dict):
            self.db_descriptors = data.get("descriptors", data.get("boq_descriptors", list(data.values())[0]))
        else:
            self.db_descriptors = data
        self.db_descriptors = F.normalize(self.db_descriptors.float().cuda(), dim=1)
        self.n_patches = self.db_descriptors.shape[0]

        # Optional PCA dimensionality reduction (VLAD paths only)
        # For use_vlad_trt: PCA is baked into the engine for queries,
        # but we still need to project the DB. Set _pca_components=None
        # so coarse_match() does NOT re-apply PCA on query descriptors.
        self._pca_mean: Optional[torch.Tensor] = None
        self._pca_components: Optional[torch.Tensor] = None
        pca_dim = config.get("vlad_pca_dim", 0)
        if (use_vlad or use_vlad_trt) and pca_dim > 0:
            pca_path = db_path.parent / "pca_components.pt"
            if pca_path.exists():
                pca_data = torch.load(str(pca_path), map_location="cuda")
                self._pca_mean = pca_data["mean"].cuda()
                self._pca_components = pca_data["components"][:pca_dim].cuda()
                k = self._pca_components.shape[0]
                orig_dim = self.db_descriptors.shape[1]
                self.db_descriptors = F.normalize(
                    (self.db_descriptors - self._pca_mean) @ self._pca_components.T, dim=1)
                print(f"PCA (pre-computed): {orig_dim}→{k} dims, "
                      f"trained on {pca_data.get('n_train_samples', '?')} samples, "
                      f"DB: {self.db_descriptors.shape}")
            else:
                orig_dim = self.db_descriptors.shape[1]
                mean = self.db_descriptors.mean(dim=0)
                X_c = self.db_descriptors - mean
                _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
                k = min(pca_dim, Vh.shape[0])
                explained = (S[:k] ** 2).sum() / (S ** 2).sum()
                self._pca_mean = mean
                self._pca_components = Vh[:k]
                self.db_descriptors = F.normalize(X_c @ self._pca_components.T, dim=1)
                print(f"PCA (on-the-fly): {orig_dim}→{k} dims, explained: {explained:.1%}, "
                      f"DB: {self.db_descriptors.shape}")

        # For VLAD TRT: DB is projected, but engine already outputs PCA'd descriptors
        # so clear _pca_components to prevent coarse_match() from re-applying PCA.
        if use_vlad_trt:
            self._pca_mean = None
            self._pca_components = None

        # Patch names
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

        # ── Load match module (needed for fine matcher and optional BoQ) ─
        mod = _import_match_module(config["script_dir"])

        # ── Coarse matcher: BoQ TRT (only when not using VLAD or VLAD TRT) ─
        if not use_vlad and not use_vlad_trt:
            boq_path = script_dir / config["boq_engine_path"]
            print(f"Loading coarse BoQ engine: {boq_path}")
            self.extractor = mod.DINOv3LoRABoQExtractorTRT(
                engine_path=str(boq_path),
                image_size=self.image_size,
                grayscale=self.grayscale,
            )

        # ── Fine matcher (TRT or PyTorch) ────────────────────────────────
        use_pytorch = config.get("use_pytorch_matcher", False)
        if use_pytorch:
            fine_ckpt = script_dir / config["fine_ckpt_path"]
            self.matcher = mod.PyTorchMatcherWrapper(str(fine_ckpt), input_size=self.matcher_resolution)
            print(f"Fine matcher: PyTorch ELoFTR (FP32) from {fine_ckpt}")
        else:
            fine_path = script_dir / config["fine_engine_path"]
            self.matcher = mod.TRTWrapper(str(fine_path))
            print(f"Fine matcher: TensorRT ELoFTR from {fine_path}")
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

        # Apply PCA projection if enabled
        if self._pca_components is not None:
            desc = F.normalize((desc - self._pca_mean) @ self._pca_components.T, dim=1)

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

    def compute_similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Cosine similarity between two images using the coarse descriptor extractor."""
        t_a = self.extractor.preprocess(img_a)
        t_b = self.extractor.preprocess(img_b)
        d_a = F.normalize(self.extractor(t_a).float(), dim=1)
        d_b = F.normalize(self.extractor(t_b).float(), dim=1)
        if self._pca_components is not None:
            d_a = F.normalize((d_a - self._pca_mean) @ self._pca_components.T, dim=1)
            d_b = F.normalize((d_b - self._pca_mean) @ self._pca_components.T, dim=1)
        return float((d_a @ d_b.T).squeeze())

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
        n_total_matches = int(mask.sum())
        if n_total_matches < self.min_inliers_ransac:
            return None

        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]

        # Compute flow geometry metrics on pre-RANSAC confidence-filtered matches
        flow_metrics = compute_flow_metrics(mkpts0, mkpts1)

        # Scale patch keypoints to composite pixel space
        mkpts1_patch = mkpts1 * np.array([[patch_w / res, patch_h / res]], dtype=np.float32)

        # --- PnP ---
        lat, lon, inliers, method, heading_deg = None, None, 0, "homography", None

        if len(mkpts0) >= self.min_inliers_ransac:
            pnp_result = self._pnp_refine_gps(mkpts0, mkpts1_patch, flat_meta)
            if pnp_result is not None:
                lat, lon, heading_deg, inliers = pnp_result
                method = "pnp"

        # --- Homography fallback ---
        vis_mask = None
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
            vis_mask = hm.ravel().astype(bool)
        else:
            # PnP: compute homography just for inlier mask (visualization only)
            if len(mkpts0) >= 4:
                _, hm = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
                if hm is not None:
                    vis_mask = hm.ravel().astype(bool)

        if inliers < self.min_inliers:
            return None

        # Filter to inliers only for visualization
        vis0 = mkpts0[vis_mask] if vis_mask is not None else mkpts0
        vis1 = mkpts1_patch[vis_mask] if vis_mask is not None else mkpts1_patch

        return FineResult(
            lat=lat, lon=lon, inliers=inliers, method=method,
            heading_deg=heading_deg, patch_name=patch_name,
            mkpts_drone=vis0, mkpts_patch=vis1,
            H=None,
            flow_consistency=flow_metrics["flow_consistency"],
            flow_magnitude_cv=flow_metrics["flow_magnitude_cv"],
            inlier_ratio=inliers / max(n_total_matches, 1),
            flow_heading_deg=flow_metrics["flow_heading_deg"],
            n_total_matches=n_total_matches,
        )

    def fine_match_on_mosaic(
        self,
        frame_bgr: np.ndarray,
        mosaic_rotated: np.ndarray,
        mosaic_meta: dict,
        rot_crop_M_inv: np.ndarray,
    ) -> Optional[FineResult]:
        """Fine match drone frame against a heading-aligned square mosaic.

        mosaic_rotated: square, heading-aligned crop (no black regions).
        rot_crop_M_inv: [2,3] affine that maps mosaic_rotated pixel coords
            back to the North-up mosaic pixel coords for GPS conversion.
        mosaic_meta: GPS bounds and pixel size of the North-up mosaic.
        """
        res = self.matcher_resolution
        mh, mw = mosaic_rotated.shape[:2]

        frame_cropped = _center_crop_square(frame_bgr)
        q_gray = cv2.cvtColor(
            cv2.resize(frame_cropped, (res, res)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32) / 255.0
        p_gray = cv2.cvtColor(
            cv2.resize(mosaic_rotated, (res, res)), cv2.COLOR_BGR2GRAY
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
        n_total_matches = int(mask.sum())
        if n_total_matches < self.min_inliers_ransac:
            return None

        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]

        # Compute flow geometry metrics on pre-RANSAC confidence-filtered matches
        flow_metrics = compute_flow_metrics(mkpts0, mkpts1)

        # Scale keypoints from matcher res to mosaic_rotated pixel space
        mkpts1_rot = mkpts1 * np.array([[mw / res, mh / res]], dtype=np.float32)

        # Map back to North-up mosaic pixel space via inverse affine
        ones = np.ones((len(mkpts1_rot), 1), dtype=np.float32)
        pts_h = np.hstack([mkpts1_rot, ones])  # [N, 3]
        mkpts1_northup = (rot_crop_M_inv @ pts_h.T).T.astype(np.float32)  # [N, 2]

        # flat_meta uses the North-up mosaic dimensions for GPS conversion
        northup_h = mosaic_meta["h"]
        northup_w = mosaic_meta["w"]
        flat_meta = {
            "min_lat": mosaic_meta["min_lat"],
            "max_lat": mosaic_meta["max_lat"],
            "min_lon": mosaic_meta["min_lon"],
            "max_lon": mosaic_meta["max_lon"],
            "patch_h": northup_h,
            "patch_w": northup_w,
        }

        lat, lon, inliers, method, heading_deg_out = None, None, 0, "homography", None

        if len(mkpts0) >= self.min_inliers_ransac:
            pnp_result = self._pnp_refine_gps(mkpts0, mkpts1_northup, flat_meta)
            if pnp_result is not None:
                lat, lon, heading_deg_out, inliers = pnp_result
                method = "pnp"

        vis_mask = None
        if lat is None:
            if len(mkpts0) < 4:
                return None
            # Homography: scale drone kpts to North-up mosaic space, match against North-up patch kpts
            mkpts0_scaled = mkpts0 * np.array([[northup_w / res, northup_h / res]], dtype=np.float32)
            H_northup, hm = cv2.findHomography(
                mkpts0_scaled, mkpts1_northup, cv2.RANSAC, 5.0,
            )
            if H_northup is None:
                return None
            inliers = int(hm.sum())
            if inliers < self.min_inliers_ransac:
                return None
            # Map drone center through homography to get GPS
            cx_d = northup_w / 2.0
            cy_d = northup_h / 2.0
            pt = np.array([[[cx_d, cy_d]]], dtype=np.float32)
            pt_res = cv2.perspectiveTransform(pt, H_northup)[0][0]
            px = max(0.0, min(float(northup_w - 1), float(pt_res[0])))
            py = max(0.0, min(float(northup_h - 1), float(pt_res[1])))
            lat = flat_meta["max_lat"] - py * (flat_meta["max_lat"] - flat_meta["min_lat"]) / (northup_h - 1)
            lon = flat_meta["min_lon"] + px * (flat_meta["max_lon"] - flat_meta["min_lon"]) / (northup_w - 1)
            method = "homography"
            vis_mask = hm.ravel().astype(bool)
        else:
            if len(mkpts0) >= 4:
                _, hm = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
                if hm is not None:
                    vis_mask = hm.ravel().astype(bool)

        if inliers < self.min_inliers:
            return None

        if vis_mask is not None:
            vis0 = mkpts0[vis_mask]
            vis1 = mkpts1_rot[vis_mask]
        else:
            vis0 = mkpts0
            vis1 = mkpts1_rot
            if len(vis0) > 80:
                idx = np.random.choice(len(vis0), 80, replace=False)
                vis0 = vis0[idx]
                vis1 = vis1[idx]

        return FineResult(
            lat=lat, lon=lon, inliers=inliers, method=method,
            heading_deg=heading_deg_out, patch_name="mosaic",
            mkpts_drone=vis0, mkpts_patch=vis1,
            H=None,
            flow_consistency=flow_metrics["flow_consistency"],
            flow_magnitude_cv=flow_metrics["flow_magnitude_cv"],
            inlier_ratio=inliers / max(n_total_matches, 1),
            flow_heading_deg=flow_metrics["flow_heading_deg"],
            n_total_matches=n_total_matches,
        )

    def fine_match_on_satellite(
        self,
        frame_bgr: np.ndarray,
        satellite_crop: np.ndarray,
        mosaic_meta: dict,
        warp_M_inv: np.ndarray,
    ) -> Optional[FineResult]:
        """Fine match drone frame against the perspective-warped satellite crop.

        The satellite_crop is already aligned to the drone FOV orientation.
        warp_M_inv maps satellite_crop pixel coords back to North-up mosaic
        pixel coords for GPS conversion.
        """
        res = self.matcher_resolution
        sh, sw = satellite_crop.shape[:2]

        frame_cropped = _center_crop_square(frame_bgr)
        q_gray = cv2.cvtColor(
            cv2.resize(frame_cropped, (res, res)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32) / 255.0
        p_gray = cv2.cvtColor(
            cv2.resize(satellite_crop, (res, res)), cv2.COLOR_BGR2GRAY
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
        n_total_matches = int(mask.sum())
        if n_total_matches < self.min_inliers_ransac:
            return None

        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]

        # Compute flow geometry metrics on pre-RANSAC confidence-filtered matches
        flow_metrics = compute_flow_metrics(mkpts0, mkpts1)

        # Scale keypoints from matcher res to satellite_crop pixel space
        mkpts1_sat = mkpts1 * np.array([[sw / res, sh / res]], dtype=np.float32)

        # Un-warp: satellite_crop pixels → North-up mosaic pixels via M_inv
        pts_h = np.hstack([mkpts1_sat, np.ones((len(mkpts1_sat), 1), dtype=np.float32)])
        mosaic_pts_h = (warp_M_inv @ pts_h.T).T  # [N, 3]
        mosaic_pts = mosaic_pts_h[:, :2] / mosaic_pts_h[:, 2:3]  # dehomogenize
        mkpts1_northup = mosaic_pts.astype(np.float32)

        mh = mosaic_meta["h"]
        mw = mosaic_meta["w"]
        flat_meta = {
            "min_lat": mosaic_meta["min_lat"],
            "max_lat": mosaic_meta["max_lat"],
            "min_lon": mosaic_meta["min_lon"],
            "max_lon": mosaic_meta["max_lon"],
            "patch_h": mh,
            "patch_w": mw,
        }

        lat, lon, inliers, method, heading_deg = None, None, 0, "homography", None

        if len(mkpts0) >= self.min_inliers_ransac:
            pnp_result = self._pnp_refine_gps(mkpts0, mkpts1_northup, flat_meta)
            if pnp_result is not None:
                lat, lon, heading_deg, inliers = pnp_result
                method = "pnp"

        vis_mask = None
        if lat is None:
            if len(mkpts0) < 4:
                return None
            H_northup, hm = cv2.findHomography(
                mkpts0 * np.array([[mw / res, mh / res]], dtype=np.float32),
                mkpts1_northup, cv2.RANSAC, 5.0,
            )
            if H_northup is None:
                return None
            inliers = int(hm.sum())
            if inliers < self.min_inliers_ransac:
                return None
            cx_d, cy_d = mw / 2.0, mh / 2.0
            pt = np.array([[[cx_d, cy_d]]], dtype=np.float32)
            pt_res = cv2.perspectiveTransform(pt, H_northup)[0][0]
            px = max(0.0, min(float(mw - 1), float(pt_res[0])))
            py = max(0.0, min(float(mh - 1), float(pt_res[1])))
            lat = flat_meta["max_lat"] - py * (flat_meta["max_lat"] - flat_meta["min_lat"]) / (mh - 1)
            lon = flat_meta["min_lon"] + px * (flat_meta["max_lon"] - flat_meta["min_lon"]) / (mw - 1)
            method = "homography"
            vis_mask = hm.ravel().astype(bool)
        else:
            # PnP succeeded: compute homography in 320px space just for viz inlier mask
            if len(mkpts0) >= 4:
                _, hm = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
                if hm is not None:
                    vis_mask = hm.ravel().astype(bool)
            # Fallback: if homography degenerate, use conf-filtered matches as-is

        if inliers < self.min_inliers:
            return None

        if vis_mask is not None:
            vis0 = mkpts0[vis_mask]
            vis1 = mkpts1_sat[vis_mask]
        else:
            vis0 = mkpts0
            vis1 = mkpts1_sat
            if len(vis0) > 80:
                idx = np.random.choice(len(vis0), 80, replace=False)
                vis0 = vis0[idx]
                vis1 = vis1[idx]

        return FineResult(
            lat=lat, lon=lon, inliers=inliers, method=method,
            heading_deg=heading_deg, patch_name="satellite",
            mkpts_drone=vis0, mkpts_patch=vis1,
            H=None,
            flow_consistency=flow_metrics["flow_consistency"],
            flow_magnitude_cv=flow_metrics["flow_magnitude_cv"],
            inlier_ratio=inliers / max(n_total_matches, 1),
            flow_heading_deg=flow_metrics["flow_heading_deg"],
            n_total_matches=n_total_matches,
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
        print(f"[PnP] entry: {len(mkpts_drone)} pts, alt={self.altitude_m:.1f}m, "
              f"patch=({meta['patch_w']}x{meta['patch_h']})")
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
        except Exception as e:
            print(f"[PnP] solvePnPRansac exception: {e}")
            return None

        if not ok or inlier_idx is None or len(inlier_idx) < self.min_inliers_ransac:
            n_inl = len(inlier_idx) if inlier_idx is not None else 0
            print(f"[PnP] RANSAC failed: ok={ok}, inliers={n_inl}, min={self.min_inliers_ransac}")
            return None

        # RANSAC can return ok=True with NaN for coplanar scenes — re-solve with IPPE
        if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
            inlier_obj = obj_pts[inlier_idx.flatten()]
            inlier_img = img_pts[inlier_idx.flatten()]
            try:
                ok_ip, rvec_ip, tvec_ip = cv2.solvePnP(
                    inlier_obj, inlier_img, K, None,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                if ok_ip and np.all(np.isfinite(rvec_ip)) and np.all(np.isfinite(tvec_ip)):
                    rvec, tvec = rvec_ip, tvec_ip
                else:
                    print(f"[PnP] IPPE also failed")
                    return None
            except Exception as e:
                print(f"[PnP] IPPE exception: {e}")
                return None

        # Refine pose using inliers only (fall back to RANSAC result if refinement diverges)
        rvec_ref, tvec_ref = rvec.copy(), tvec.copy()
        inlier_obj = obj_pts[inlier_idx.flatten()]
        inlier_img = img_pts[inlier_idx.flatten()]
        try:
            ok2, rvec2, tvec2 = cv2.solvePnP(
                inlier_obj, inlier_img, K, None, rvec_ref, tvec_ref, True,
                cv2.SOLVEPNP_ITERATIVE,
            )
            if ok2 and np.all(np.isfinite(tvec2)) and np.all(np.isfinite(rvec2)):
                rvec, tvec = rvec2, tvec2
        except Exception:
            pass  # keep RANSAC result

        R, _ = cv2.Rodrigues(rvec)
        cam_pos = (-R.T @ tvec).flatten()

        if not np.all(np.isfinite(cam_pos)):
            det = np.linalg.det(R)
            obj_span = obj_pts.max(axis=0) - obj_pts.min(axis=0)
            img_span = img_pts.max(axis=0) - img_pts.min(axis=0)
            print(f"[PnP] non-finite cam_pos: rvec={rvec.flatten()}, tvec={tvec.flatten()}, "
                  f"det(R)={det:.3f}, obj_span={obj_span}, img_span={img_span}, "
                  f"inliers={len(inlier_idx)}")
            return None

        # Sanity check: PnP altitude must be within 50% of live altimeter reading
        pnp_alt = float(cam_pos[2])
        print(f"[PnP] alt_est={pnp_alt:.1f}m  alt_live={self.altitude_m:.1f}m  "
              f"err={abs(pnp_alt - self.altitude_m):.1f}m  inliers={len(inlier_idx)}")
        if self.altitude_m > 0.0 and abs(pnp_alt - self.altitude_m) > self.altitude_m * 0.5:
            print(f"[PnP] REJECTED (>{self.altitude_m * 0.5:.1f}m threshold)")
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
