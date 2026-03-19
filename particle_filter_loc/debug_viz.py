"""Debug visualization for the particle filter geo-localization pipeline."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .particle_filter import ParticleFilter, Phase

# Phase colors (BGR)
_PHASE_COLOR = {
    Phase.UNINIT:     (128, 128, 128),
    Phase.DISPERSED:  (0,   165, 255),   # orange
    Phase.CONVERGING: (0,   255, 255),   # yellow
    Phase.TRACKING:   (0,   200, 0),     # green
}

_SZ = 320          # all panels are 320x320
_GAP = 4
_STATUS_H = 48

# Minimap path colors (BGR)
_COLOR_GT  = (0,   50, 220)
_COLOR_PF  = (220, 80,  0)
_COLOR_GT_CUR = (0, 0, 255)
_COLOR_PF_CUR = (255, 120, 0)


def _blank():
    return np.zeros((_SZ, _SZ, 3), dtype=np.uint8)


def _resize(img):
    if img is None:
        return _blank()
    return cv2.resize(img, (_SZ, _SZ), interpolation=cv2.INTER_AREA)


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == h:
        return frame
    s = min(w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return frame[y0:y0+s, x0:x0+s]


def _text(img, lines, x=8, y0=20, scale=0.5, color=(255, 255, 255), thickness=1):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)


def _build_match_panel(
    drone_320: np.ndarray,
    ref_img: Optional[np.ndarray],
    mkpts_drone: Optional[np.ndarray],
    mkpts_ref: Optional[np.ndarray],
    label: str,
    method: str = "",
    inliers: int = 0,
    line_color: Tuple[int, int, int] = (0, 220, 100),
) -> np.ndarray:
    """Build a 640x320 match panel: [drone | ref] with correspondence lines."""
    p_drone = drone_320.copy()
    p_ref = _resize(ref_img)

    has_matches = (mkpts_drone is not None and mkpts_ref is not None
                   and len(mkpts_drone) > 0 and len(mkpts_ref) > 0)

    # Keypoints are in matcher space (320x320). If ref_img is not 320x320,
    # scale ref keypoints to 320x320.
    if has_matches:
        ref_h = ref_img.shape[0] if ref_img is not None else _SZ
        ref_w = ref_img.shape[1] if ref_img is not None else _SZ
        sx_r = _SZ / ref_w
        sy_r = _SZ / ref_h

        # Draw keypoints
        for pt in mkpts_drone:
            cv2.circle(p_drone, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)
        for pt in mkpts_ref:
            cv2.circle(p_ref, (int(pt[0] * sx_r), int(pt[1] * sy_r)), 2, (0, 100, 255), -1, cv2.LINE_AA)

    # Combine side by side
    panel = np.hstack([p_drone, p_ref])

    # Draw match lines across the two halves
    if has_matches:
        n_draw = min(len(mkpts_drone), 60)
        for i in range(n_draw):
            px_d = int(mkpts_drone[i, 0])
            py_d = int(mkpts_drone[i, 1])
            px_r = int(mkpts_ref[i, 0] * sx_r) + _SZ
            py_r = int(mkpts_ref[i, 1] * sy_r)
            px_d = np.clip(px_d, 0, _SZ - 1)
            py_d = np.clip(py_d, 0, _SZ - 1)
            px_r = np.clip(px_r, _SZ, _SZ * 2 - 1)
            py_r = np.clip(py_r, 0, _SZ - 1)
            cv2.line(panel, (px_d, py_d), (px_r, py_r), line_color, 1, cv2.LINE_AA)

        info = f"{method} {inliers}inl"
        _text(panel, [info], x=8, y0=20, scale=0.4, color=(0, 255, 0))
    else:
        _text(panel, ["no match"], x=8, y0=20, scale=0.4, color=(100, 100, 100))

    # Labels
    cv2.putText(panel, "DRONE", (4, _SZ - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, label, (_SZ + 4, _SZ - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return panel


def _build_minimap(
    gt_path:  List[Tuple[float, float]],
    pf_path:  List[Tuple[float, float]],
    phase_path: List[Phase],
) -> np.ndarray:
    """Draw accumulated RTK (red) and PF (blue) trajectories on a dark canvas."""
    w = h = _SZ
    panel = _blank()

    all_pts = gt_path + pf_path
    if len(all_pts) < 2:
        _text(panel, ["minimap: no path yet"])
        cv2.putText(panel, "MINIMAP", (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return panel

    all_e = [p[0] for p in all_pts]
    all_n = [p[1] for p in all_pts]
    min_e, max_e = min(all_e), max(all_e)
    min_n, max_n = min(all_n), max(all_n)

    pad_e = max((max_e - min_e) * 0.1, 5.0)
    pad_n = max((max_n - min_n) * 0.1, 5.0)
    min_e -= pad_e;  max_e += pad_e
    min_n -= pad_n;  max_n += pad_n
    span_e = max_e - min_e
    span_n = max_n - min_n

    def to_px(e, n):
        px = int((e - min_e) / span_e * (w - 1))
        py = int((1.0 - (n - min_n) / span_n) * (h - 1))
        return np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)

    for frac in [0.25, 0.5, 0.75]:
        gx, gy = int(frac * w), int(frac * h)
        cv2.line(panel, (gx, 0), (gx, h), (35, 35, 35), 1)
        cv2.line(panel, (0, gy), (w, gy), (35, 35, 35), 1)

    if len(gt_path) >= 2:
        pts_px = [to_px(e, n) for e, n in gt_path]
        for i in range(1, len(pts_px)):
            cv2.line(panel, pts_px[i - 1], pts_px[i], _COLOR_GT, 1, cv2.LINE_AA)

    if len(pf_path) >= 2:
        pts_px = [to_px(e, n) for e, n in pf_path]
        for i in range(1, len(pts_px)):
            seg_color = _PHASE_COLOR.get(phase_path[i], _COLOR_PF)
            cv2.line(panel, pts_px[i - 1], pts_px[i], seg_color, 2, cv2.LINE_AA)

    if gt_path:
        cx, cy = to_px(*gt_path[-1])
        cv2.circle(panel, (cx, cy), 5, _COLOR_GT_CUR, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)
    if pf_path:
        cx, cy = to_px(*pf_path[-1])
        cv2.circle(panel, (cx, cy), 5, _COLOR_PF_CUR, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.line(panel, (8, h - 28), (24, h - 28), _COLOR_GT_CUR, 2)
    cv2.line(panel, (8, h - 14), (24, h - 14), _COLOR_PF_CUR, 2)
    _text(panel, ["RTK", "PF"], x=28, y0=h - 34, scale=0.38, color=(200, 200, 200))

    metres_per_px = span_e / w
    target_bar_m = max(round(span_e / 4 / 10) * 10, 5.0)
    bar_px = max(int(target_bar_m / metres_per_px), 4)
    bx0, by0 = w - bar_px - 10, 10
    cv2.line(panel, (bx0, by0), (bx0 + bar_px, by0), (160, 160, 160), 2)
    cv2.line(panel, (bx0, by0 - 3), (bx0, by0 + 3), (160, 160, 160), 1)
    cv2.line(panel, (bx0 + bar_px, by0 - 3), (bx0 + bar_px, by0 + 3), (160, 160, 160), 1)
    _text(panel, [f"{int(target_bar_m)}m"], bx0, by0 + 14, scale=0.38, color=(160, 160, 160))

    cv2.putText(panel, "MINIMAP", (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def _build_candidates_row(
    names: List[str],
    sims: List[float],
    patches: List[Optional[np.ndarray]],
    fine_matched_name: str = "",
    total_w: int = 1288,
) -> np.ndarray:
    """Horizontal row of K candidate thumbnails with rank and sim score."""
    thumb_sz = 256
    h = thumb_sz + 36
    row = np.full((h, total_w, 3), 20, dtype=np.uint8)
    k = len(names)
    if k == 0:
        _text(row, ["no coarse candidates"], x=8, y0=h // 2, color=(160, 160, 160))
        return row

    cell_w = total_w // k
    for i, (name, sim, patch) in enumerate(zip(names, sims, patches)):
        x0 = i * cell_w
        x1 = x0 + cell_w - 2
        matched = bool(name == fine_matched_name and fine_matched_name)

        cx = x0 + (cell_w - thumb_sz) // 2
        if patch is not None:
            thumb = cv2.resize(patch, (thumb_sz, thumb_sz), interpolation=cv2.INTER_AREA)
            row[0:thumb_sz, cx:cx + thumb_sz] = thumb
        else:
            cv2.rectangle(row, (cx, 0), (cx + thumb_sz - 1, thumb_sz - 1), (60, 60, 60), -1)

        if sim >= 0.5:
            sim_color = (50, 210, 50)
        elif sim >= 0.3:
            sim_color = (50, 210, 210)
        else:
            sim_color = (80, 80, 220)

        _text(row, [f"#{i+1}  {sim:.3f}"], x=x0+4, y0=thumb_sz+14, scale=0.45, color=sim_color)
        short = name[-24:] if name else "—"
        _text(row, [short], x=x0+4, y0=thumb_sz+28, scale=0.35, color=(180, 180, 180))

        border_color = (0, 230, 80) if matched else (70, 70, 70)
        border_thick = 2 if matched else 1
        cv2.rectangle(row, (x0, 0), (x1 - 1, h - 1), border_color, border_thick)

    return row


class DebugVisualizer:
    """Wraps debug frame assembly, accumulates trajectory history, handles display."""

    def __init__(
        self,
        show_window: bool = True,
        window_name: str = "PF Geo-Loc Debug",
        save_frames: bool = False,
        save_dir: Optional[str] = None,
        ros_publisher=None,
        cv_bridge=None,
        max_path_len: int = 2000,
    ):
        self.show_window = show_window
        self.window_name = window_name
        self.save_frames = save_frames
        self.save_dir = save_dir
        self.ros_pub = ros_publisher
        self.bridge = cv_bridge
        self.max_path_len = max_path_len
        self._frame_idx = 0

        # Coarse match results
        self.coarse_patch: Optional[np.ndarray] = None
        self.coarse_name: str = ""
        self.coarse_sim: float = 0.0
        self.coarse_top_k_names: List[str] = []
        self.coarse_top_k_sims: List[float] = []
        self.coarse_top_k_patches: List[Optional[np.ndarray]] = []
        self.fine_matched_name: str = ""

        # Legacy fields for ros2_pf_node compatibility
        self.mkpts_drone: Optional[np.ndarray] = None
        self.mkpts_patch: Optional[np.ndarray] = None
        self.fine_H: Optional[np.ndarray] = None
        self.fine_method: str = ""
        self.fine_inliers: int = 0

        # Mosaic fine match (drone vs reconstructed satellite)
        self.mosaic_mkpts_drone: Optional[np.ndarray] = None
        self.mosaic_mkpts_ref: Optional[np.ndarray] = None
        self.mosaic_ref_img: Optional[np.ndarray] = None
        self.mosaic_fine_method: str = ""
        self.mosaic_fine_inliers: int = 0

        # Top-1 fine match (drone vs coarse top-1 patch)
        self.top1_mkpts_drone: Optional[np.ndarray] = None
        self.top1_mkpts_ref: Optional[np.ndarray] = None
        self.top1_ref_img: Optional[np.ndarray] = None
        self.top1_fine_method: str = ""
        self.top1_fine_inliers: int = 0

        # Satellite footprint
        self.satellite_footprint: Optional[np.ndarray] = None
        self.footprint_confidence: float = 0.0
        self.footprint_info_str: str = ""

        # Accumulated trajectory history
        self._gt_path:  List[Tuple[float, float]] = []
        self._pf_path:  List[Tuple[float, float]] = []
        self._phase_path: List[Phase] = []

        # Total width: row1 = 2 match panels (640 each) + gap = 1284
        self._total_w = _SZ * 2 * 2 + _GAP

        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self._total_w, _SZ * 2 + 292 + _STATUS_H + _GAP * 3)
            cv2.moveWindow(window_name, 80, 80)

        if save_frames and save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)

    def clear_fine(self):
        """Reset fine match state for both mosaic and top1."""
        self.mosaic_mkpts_drone = None
        self.mosaic_mkpts_ref = None
        self.mosaic_ref_img = None
        self.mosaic_fine_method = ""
        self.mosaic_fine_inliers = 0
        self.top1_mkpts_drone = None
        self.top1_mkpts_ref = None
        self.top1_ref_img = None
        self.top1_fine_method = ""
        self.top1_fine_inliers = 0

    def update(
        self,
        pf: ParticleFilter,
        drone_frame: np.ndarray,
        error_m: float,
        elapsed_s: float,
        gt_east: Optional[float] = None,
        gt_north: Optional[float] = None,
    ) -> np.ndarray:
        # Accumulate paths
        if gt_east is not None and gt_north is not None:
            self._gt_path.append((gt_east, gt_north))
            if len(self._gt_path) > self.max_path_len:
                self._gt_path = self._gt_path[-self.max_path_len:]

        if pf.particles is not None:
            est_e, est_n, _ = pf.estimate()
            self._pf_path.append((est_e, est_n))
            self._phase_path.append(pf.phase)
            if len(self._pf_path) > self.max_path_len:
                self._pf_path = self._pf_path[-self.max_path_len:]
                self._phase_path = self._phase_path[-self.max_path_len:]

        phase = pf.phase
        phase_color = _PHASE_COLOR.get(phase, (255, 255, 255))

        # Drone frame: center crop to square, resize to 320x320
        drone_320 = cv2.resize(_center_crop_square(drone_frame), (_SZ, _SZ))

        # ---- Row 1: two match panels ----
        mosaic_panel = _build_match_panel(
            drone_320, self.mosaic_ref_img,
            self.mosaic_mkpts_drone, self.mosaic_mkpts_ref,
            label="MOSAIC", method=self.mosaic_fine_method,
            inliers=self.mosaic_fine_inliers, line_color=(0, 220, 100),
        )
        top1_panel = _build_match_panel(
            drone_320, self.top1_ref_img,
            self.top1_mkpts_drone, self.top1_mkpts_ref,
            label="TOP-1", method=self.top1_fine_method,
            inliers=self.top1_fine_inliers, line_color=(220, 160, 0),
        )
        gap_v = np.zeros((_SZ, _GAP, 3), dtype=np.uint8)
        row1 = np.hstack([mosaic_panel, gap_v, top1_panel])

        # ---- Row 2: satellite + particles + minimap + pad ----
        # Satellite footprint panel
        if self.satellite_footprint is not None:
            p_sat = _resize(self.satellite_footprint)
            if self.footprint_confidence >= 0.5:
                bc = (0, 200, 0)
            elif self.footprint_confidence >= 0.3:
                bc = (0, 200, 200)
            else:
                bc = (0, 80, 200)
            cv2.rectangle(p_sat, (0, 0), (_SZ - 1, _SZ - 1), bc, 2)
            info = self.footprint_info_str or f"conf={self.footprint_confidence:.2f}"
            _text(p_sat, [info], y0=20, scale=0.4, color=bc)
            cv2.putText(p_sat, "SATELLITE", (4, _SZ - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            p_sat = _blank()
            _text(p_sat, ["no footprint"])
            cv2.putText(p_sat, "SATELLITE", (4, _SZ - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Particle scatter
        p_part = _blank()
        if pf.particles is not None and len(pf.particles) > 0:
            pts_e = pf.particles[:, 0]
            pts_n = pf.particles[:, 1]
            pad = max(pf.weighted_spread() * 3.0, 30.0)
            est_e2, est_n2, _ = pf.estimate()
            min_e2, max_e2 = est_e2 - pad, est_e2 + pad
            min_n2, max_n2 = est_n2 - pad, est_n2 + pad

            def to_px_local(e, n):
                px = int((e - min_e2) / (max_e2 - min_e2 + 1e-9) * (_SZ - 1))
                py = int((1.0 - (n - min_n2) / (max_n2 - min_n2 + 1e-9)) * (_SZ - 1))
                return np.clip(px, 0, _SZ - 1), np.clip(py, 0, _SZ - 1)

            for frac in [0.25, 0.5, 0.75]:
                cv2.line(p_part, (int(frac * _SZ), 0), (int(frac * _SZ), _SZ), (40, 40, 40), 1)
                cv2.line(p_part, (0, int(frac * _SZ)), (_SZ, int(frac * _SZ)), (40, 40, 40), 1)

            w_arr = pf.weights
            w_norm = (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-9)
            for i in range(len(pts_e)):
                px, py = to_px_local(pts_e[i], pts_n[i])
                intensity = int(w_norm[i] * 255)
                cv2.circle(p_part, (px, py), 2, (255 - intensity, 0, intensity), -1)

            ex, ey = to_px_local(est_e2, est_n2)
            cv2.drawMarker(p_part, (ex, ey), (255, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            if gt_east is not None and gt_north is not None:
                gx, gy = to_px_local(gt_east, gt_north)
                cv2.circle(p_part, (gx, gy), 5, (0, 255, 255), 2, cv2.LINE_AA)

            bar_m = 10.0
            bar_px = max(int(bar_m / (max_e2 - min_e2 + 1e-9) * _SZ), 2)
            cv2.line(p_part, (10, _SZ - 10), (10 + bar_px, _SZ - 10), (180, 180, 180), 2)
            _text(p_part, ["10m"], 10, _SZ - 22, scale=0.4, color=(180, 180, 180))

            spread = pf.weighted_spread()
            ess = pf.effective_sample_size()
            _text(p_part, [f"ESS={ess:.0f}  sprd={spread:.1f}m"], y0=20)
        else:
            _text(p_part, ["no particles"])
        cv2.putText(p_part, "PARTICLES", (4, _SZ - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Minimap
        p_mini = _build_minimap(self._gt_path, self._pf_path, self._phase_path)

        # Pad row2 to match row1 width
        row2_content = np.hstack([p_sat, gap_v, p_part, gap_v, p_mini])
        row2_w = row2_content.shape[1]
        total_w = row1.shape[1]
        if row2_w < total_w:
            pad_block = np.zeros((_SZ, total_w - row2_w, 3), dtype=np.uint8)
            row2 = np.hstack([row2_content, pad_block])
        else:
            row2 = row2_content

        # ---- Candidates row ----
        cands_row = _build_candidates_row(
            names=self.coarse_top_k_names or [],
            sims=self.coarse_top_k_sims or [],
            patches=self.coarse_top_k_patches or [],
            fine_matched_name=self.fine_matched_name,
            total_w=total_w,
        )

        # ---- Status bar ----
        status = np.zeros((_STATUS_H, total_w, 3), dtype=np.uint8)
        cv2.rectangle(status, (0, 0), (6, _STATUS_H), phase_color, -1)
        err_str = f"{error_m:.1f}m" if error_m >= 0 else "N/A"
        _text(status,
              [f"t={elapsed_s:.1f}s  phase={phase.name}  err={err_str}  "
               f"N={len(pf.particles) if pf.particles is not None else 0}  "
               f"mosaic={self.mosaic_fine_inliers}inl  top1={self.top1_fine_inliers}inl"],
              x=14, y0=_STATUS_H // 2 + 4, scale=0.48, color=phase_color)

        # ---- Assemble ----
        gap_h = np.zeros((_GAP, total_w, 3), dtype=np.uint8)
        dbg = np.vstack([row1, gap_h, row2, gap_h, cands_row, gap_h, status])

        if self.show_window:
            cv2.imshow(self.window_name, dbg)
            cv2.waitKey(1)

        if self.save_frames and self.save_dir:
            path = f"{self.save_dir}/frame_{self._frame_idx:06d}.jpg"
            cv2.imwrite(path, dbg, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if self.ros_pub is not None and self.bridge is not None:
            img_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
            self.ros_pub.publish(img_msg)

        self._frame_idx += 1
        return dbg

    def close(self):
        if self.show_window:
            cv2.destroyWindow(self.window_name)
