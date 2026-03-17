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

_PANEL_W = 320
_PANEL_H = 320
_STATUS_H = 48
_GAP = 4

# Minimap path colors (BGR)
_COLOR_GT  = (0,   50, 220)   # red  — RTK ground truth
_COLOR_PF  = (220, 80,  0)    # blue — particle filter estimate
_COLOR_GT_CUR = (0, 0, 255)   # bright red for current GT dot
_COLOR_PF_CUR = (255, 120, 0) # bright blue for current PF dot


def _blank(h=_PANEL_H, w=_PANEL_W):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _resize(img, w=_PANEL_W, h=_PANEL_H):
    if img is None:
        return _blank(h, w)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _text(img, lines, x=8, y0=20, scale=0.5, color=(255, 255, 255), thickness=1):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)


def _build_minimap(
    gt_path:  List[Tuple[float, float]],   # [(east, north), ...]
    pf_path:  List[Tuple[float, float]],
    phase_path: List[Phase],               # one per pf_path entry
    w: int = _PANEL_W,
    h: int = _PANEL_H,
) -> np.ndarray:
    """Draw accumulated RTK (red) and PF (blue) trajectories on a dark canvas."""
    panel = _blank(h, w)

    # Combine both paths to compute bounding box
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

    # Add 10% padding
    pad_e = max((max_e - min_e) * 0.1, 5.0)
    pad_n = max((max_n - min_n) * 0.1, 5.0)
    min_e -= pad_e;  max_e += pad_e
    min_n -= pad_n;  max_n += pad_n

    span_e = max_e - min_e
    span_n = max_n - min_n

    def to_px(e: float, n: float) -> Tuple[int, int]:
        px = int((e - min_e) / span_e * (w - 1))
        py = int((1.0 - (n - min_n) / span_n) * (h - 1))
        return np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)

    # Draw light grid
    for frac in [0.25, 0.5, 0.75]:
        gx = int(frac * w)
        gy = int(frac * h)
        cv2.line(panel, (gx, 0), (gx, h), (35, 35, 35), 1)
        cv2.line(panel, (0, gy), (w, gy), (35, 35, 35), 1)

    # Draw GT path (red, thin)
    if len(gt_path) >= 2:
        pts_px = [to_px(e, n) for e, n in gt_path]
        for i in range(1, len(pts_px)):
            cv2.line(panel, pts_px[i - 1], pts_px[i], _COLOR_GT, 1, cv2.LINE_AA)

    # Draw PF path (blue), colored by phase
    if len(pf_path) >= 2:
        pts_px = [to_px(e, n) for e, n in pf_path]
        for i in range(1, len(pts_px)):
            seg_color = _PHASE_COLOR.get(phase_path[i], _COLOR_PF)
            cv2.line(panel, pts_px[i - 1], pts_px[i], seg_color, 2, cv2.LINE_AA)

    # Current position dots
    if gt_path:
        cx, cy = to_px(*gt_path[-1])
        cv2.circle(panel, (cx, cy), 5, _COLOR_GT_CUR, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

    if pf_path:
        cx, cy = to_px(*pf_path[-1])
        cv2.circle(panel, (cx, cy), 5, _COLOR_PF_CUR, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend
    cv2.line(panel, (8, h - 28), (24, h - 28), _COLOR_GT_CUR, 2)
    cv2.line(panel, (8, h - 14), (24, h - 14), _COLOR_PF_CUR, 2)
    _text(panel, ["RTK", "PF"], x=28, y0=h - 34, scale=0.38, color=(200, 200, 200))

    # Scale bar: compute a round number of metres
    metres_per_px = span_e / w
    target_bar_m = max(round(span_e / 4 / 10) * 10, 5.0)   # nearest 10m, min 5m
    bar_px = int(target_bar_m / metres_per_px)
    bar_px = max(bar_px, 4)
    bx0, by0 = w - bar_px - 10, 10
    cv2.line(panel, (bx0, by0), (bx0 + bar_px, by0), (160, 160, 160), 2)
    cv2.line(panel, (bx0, by0 - 3), (bx0, by0 + 3), (160, 160, 160), 1)
    cv2.line(panel, (bx0 + bar_px, by0 - 3), (bx0 + bar_px, by0 + 3), (160, 160, 160), 1)
    _text(panel, [f"{int(target_bar_m)}m"], bx0, by0 + 14, scale=0.38, color=(160, 160, 160))

    cv2.putText(panel, "MINIMAP", (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def build_debug_frame(
    pf: ParticleFilter,
    drone_frame: np.ndarray,
    coarse_patch: Optional[np.ndarray],
    coarse_name: str,
    coarse_sim: float,
    mkpts_drone: Optional[np.ndarray],
    mkpts_patch: Optional[np.ndarray],
    fine_method: str,
    fine_inliers: int,
    error_m: float,
    elapsed_s: float,
    gt_east: Optional[float] = None,
    gt_north: Optional[float] = None,
    gt_path: Optional[List[Tuple[float, float]]] = None,
    pf_path: Optional[List[Tuple[float, float]]] = None,
    phase_path: Optional[List[Phase]] = None,
    fine_H: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Builds a 4-panel + status-bar debug image:
      [Drone frame | Coarse patch | Particle scatter | Minimap]
      [                    Status bar                         ]
    """
    phase = pf.phase
    phase_color = _PHASE_COLOR.get(phase, (255, 255, 255))

    # ------------------------------------------------------------------ #
    # Panel 1: drone frame + fine keypoints
    # ------------------------------------------------------------------ #
    p1 = _resize(drone_frame)
    if mkpts_drone is not None and len(mkpts_drone) > 0:
        res = drone_frame.shape[1] if drone_frame is not None else 320
        sx = _PANEL_W / res
        sy = _PANEL_H / res
        for pt in mkpts_drone:
            cv2.circle(p1, (int(pt[0] * sx), int(pt[1] * sy)), 3, (0, 255, 0), -1, cv2.LINE_AA)
        _text(p1, [f"fine: {fine_method} {fine_inliers}inl"], color=(0, 255, 0))
    else:
        _text(p1, ["no fine match"])
    cv2.putText(p1, "DRONE", (4, _PANEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Panel 2: coarse patch + patch keypoints + projected drone footprint
    # ------------------------------------------------------------------ #
    p2 = _resize(coarse_patch)
    patch_h_orig = coarse_patch.shape[0] if coarse_patch is not None else _PANEL_H
    patch_w_orig = coarse_patch.shape[1] if coarse_patch is not None else _PANEL_W
    sx_p = _PANEL_W / patch_w_orig
    sy_p = _PANEL_H / patch_h_orig

    if mkpts_patch is not None and len(mkpts_patch) > 0:
        for pt in mkpts_patch:
            cv2.circle(p2, (int(pt[0] * sx_p), int(pt[1] * sy_p)), 3, (0, 100, 255), -1, cv2.LINE_AA)

    # Project drone image corners onto patch to show perspective footprint
    if fine_H is not None:
        corners = np.float32([[0, 0], [320, 0], [320, 320], [0, 320]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, fine_H)
        # fine_H maps 320×320 drone → 320×320 matcher space; scale to panel
        scale_to_panel = np.array([_PANEL_W / 320.0, _PANEL_H / 320.0])
        proj_pts = (proj.reshape(-1, 2) * scale_to_panel).astype(np.int32)
        overlay = p2.copy()
        cv2.fillPoly(overlay, [proj_pts], (40, 160, 40))
        cv2.addWeighted(overlay, 0.25, p2, 0.75, 0, p2)
        cv2.polylines(p2, [proj_pts], True, (0, 255, 160), 2, cv2.LINE_AA)

    short_name = coarse_name[-20:] if coarse_name else "—"
    _text(p2, [f"sim={coarse_sim:.3f}", short_name])
    cv2.putText(p2, "COARSE", (4, _PANEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Panel 3: particle scatter (local ENU, zoomed)
    # ------------------------------------------------------------------ #
    p3 = _blank()
    if pf.particles is not None and len(pf.particles) > 0:
        pts_e = pf.particles[:, 0]
        pts_n = pf.particles[:, 1]
        pad = max(pf.weighted_spread() * 3.0, 30.0)
        est_e, est_n, _ = pf.estimate()
        min_e2, max_e2 = est_e - pad, est_e + pad
        min_n2, max_n2 = est_n - pad, est_n + pad

        def to_px_local(e, n):
            px = int((e - min_e2) / (max_e2 - min_e2 + 1e-9) * (_PANEL_W - 1))
            py = int((1.0 - (n - min_n2) / (max_n2 - min_n2 + 1e-9)) * (_PANEL_H - 1))
            return np.clip(px, 0, _PANEL_W - 1), np.clip(py, 0, _PANEL_H - 1)

        for frac in [0.25, 0.5, 0.75]:
            cv2.line(p3, (int(frac * _PANEL_W), 0), (int(frac * _PANEL_W), _PANEL_H), (40, 40, 40), 1)
            cv2.line(p3, (0, int(frac * _PANEL_H)), (_PANEL_W, int(frac * _PANEL_H)), (40, 40, 40), 1)

        w_arr = pf.weights
        w_norm = (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-9)
        for i in range(len(pts_e)):
            px, py = to_px_local(pts_e[i], pts_n[i])
            intensity = int(w_norm[i] * 255)
            cv2.circle(p3, (px, py), 2, (255 - intensity, 0, intensity), -1)

        ex, ey = to_px_local(est_e, est_n)
        cv2.drawMarker(p3, (ex, ey), (255, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

        if gt_east is not None and gt_north is not None:
            gx, gy = to_px_local(gt_east, gt_north)
            cv2.circle(p3, (gx, gy), 5, (0, 255, 255), 2, cv2.LINE_AA)

        bar_m = 10.0
        bar_px = max(int(bar_m / (max_e2 - min_e2 + 1e-9) * _PANEL_W), 2)
        cv2.line(p3, (10, _PANEL_H - 10), (10 + bar_px, _PANEL_H - 10), (180, 180, 180), 2)
        _text(p3, ["10m"], 10, _PANEL_H - 22, scale=0.4, color=(180, 180, 180))

        spread = pf.weighted_spread()
        ess = pf.effective_sample_size()
        _text(p3, [f"ESS={ess:.0f}  sprd={spread:.1f}m"], y0=20)
    else:
        _text(p3, ["no particles"])
    cv2.putText(p3, "PARTICLES", (4, _PANEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Panel 4: minimap
    # ------------------------------------------------------------------ #
    p4 = _build_minimap(
        gt_path or [],
        pf_path or [],
        phase_path or [],
    )

    # ------------------------------------------------------------------ #
    # Status bar
    # ------------------------------------------------------------------ #
    n_panels = 4
    total_w = _PANEL_W * n_panels + _GAP * (n_panels - 1)
    status = np.zeros((_STATUS_H, total_w, 3), dtype=np.uint8)
    cv2.rectangle(status, (0, 0), (6, _STATUS_H), phase_color, -1)
    err_str = f"{error_m:.1f}m" if error_m >= 0 else "N/A"
    _text(status,
          [f"t={elapsed_s:.1f}s  phase={phase.name}  err={err_str}  "
           f"N={len(pf.particles) if pf.particles is not None else 0}"],
          x=14, y0=_STATUS_H // 2 + 4, scale=0.52, color=phase_color)

    # ------------------------------------------------------------------ #
    # Assemble
    # ------------------------------------------------------------------ #
    gap = np.zeros((_PANEL_H, _GAP, 3), dtype=np.uint8)
    top_row = np.hstack([p1, gap, p2, gap, p3, gap, p4])

    # Draw correspondence lines between drone panel and patch panel
    if mkpts_drone is not None and mkpts_patch is not None and len(mkpts_drone) > 0:
        p2_x0 = _PANEL_W + _GAP  # x offset of patch panel in top_row
        n_draw = min(len(mkpts_drone), 60)
        for i in range(n_draw):
            px_d = int(mkpts_drone[i, 0] * _PANEL_W / 320)
            py_d = int(mkpts_drone[i, 1] * _PANEL_H / 320)
            px_p = int(mkpts_patch[i, 0] * sx_p) + p2_x0
            py_p = int(mkpts_patch[i, 1] * sy_p)
            px_d = np.clip(px_d, 0, _PANEL_W - 1)
            py_d = np.clip(py_d, 0, _PANEL_H - 1)
            px_p = np.clip(px_p, p2_x0, p2_x0 + _PANEL_W - 1)
            py_p = np.clip(py_p, 0, _PANEL_H - 1)
            cv2.line(top_row, (px_d, py_d), (px_p, py_p), (0, 220, 100), 1, cv2.LINE_AA)

    return np.vstack([top_row, status])


class DebugVisualizer:
    """Wraps build_debug_frame, accumulates trajectory history, handles display/publish."""

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

        # Cached per-frame match results
        self.coarse_patch: Optional[np.ndarray] = None
        self.coarse_name: str = ""
        self.coarse_sim: float = 0.0
        self.mkpts_drone: Optional[np.ndarray] = None
        self.mkpts_patch: Optional[np.ndarray] = None
        self.fine_H: Optional[np.ndarray] = None
        self.fine_method: str = ""
        self.fine_inliers: int = 0

        # Accumulated trajectory history
        self._gt_path:  List[Tuple[float, float]] = []   # (east, north)
        self._pf_path:  List[Tuple[float, float]] = []
        self._phase_path: List[Phase] = []

        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            total_w = _PANEL_W * 4 + _GAP * 3
            cv2.resizeWindow(window_name, total_w, _PANEL_H + _STATUS_H)
            cv2.moveWindow(window_name, 80, 80)

        if save_frames and save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)

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

        dbg = build_debug_frame(
            pf=pf,
            drone_frame=drone_frame,
            coarse_patch=self.coarse_patch,
            coarse_name=self.coarse_name,
            coarse_sim=self.coarse_sim,
            mkpts_drone=self.mkpts_drone,
            mkpts_patch=self.mkpts_patch,
            fine_method=self.fine_method,
            fine_inliers=self.fine_inliers,
            error_m=error_m,
            elapsed_s=elapsed_s,
            gt_east=gt_east,
            gt_north=gt_north,
            gt_path=self._gt_path,
            pf_path=self._pf_path,
            phase_path=self._phase_path,
            fine_H=self.fine_H,
        )

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
