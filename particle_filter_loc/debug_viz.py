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


def build_debug_frame(
    pf: ParticleFilter,
    drone_frame: np.ndarray,
    coarse_patch: Optional[np.ndarray],
    coarse_name: str,
    coarse_sim: float,
    mkpts_drone: Optional[np.ndarray],  # [N,2] in matcher-res space
    mkpts_patch: Optional[np.ndarray],  # [N,2] in matcher-res space
    fine_method: str,
    fine_inliers: int,
    error_m: float,
    elapsed_s: float,
    gt_east: Optional[float] = None,
    gt_north: Optional[float] = None,
) -> np.ndarray:
    """
    Builds a 3-panel + status-bar debug image:
      [Drone frame | Coarse patch | Particle scatter]
      [           Status bar                        ]
    """
    phase = pf.phase
    phase_color = _PHASE_COLOR.get(phase, (255, 255, 255))

    # ------------------------------------------------------------------ #
    # Panel 1: drone frame + fine keypoints
    # ------------------------------------------------------------------ #
    p1 = _resize(drone_frame)
    if mkpts_drone is not None and len(mkpts_drone) > 0:
        # Scale keypoints from matcher-res to panel size
        if drone_frame is not None:
            res = drone_frame.shape[1]  # assume square
        else:
            res = 320
        scale_x = _PANEL_W / res
        scale_y = _PANEL_H / res
        for pt in mkpts_drone:
            cx = int(pt[0] * scale_x)
            cy = int(pt[1] * scale_y)
            cv2.circle(p1, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)
        _text(p1, [f"fine: {fine_method} {fine_inliers}inl"], color=(0, 255, 0))
    else:
        _text(p1, ["no fine match"])

    cv2.putText(p1, "DRONE", (4, _PANEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Panel 2: coarse patch + patch keypoints
    # ------------------------------------------------------------------ #
    p2 = _resize(coarse_patch)
    if mkpts_patch is not None and len(mkpts_patch) > 0:
        res = 320  # matcher res
        scale_x = _PANEL_W / res
        scale_y = _PANEL_H / res
        for pt in mkpts_patch:
            cx = int(pt[0] * scale_x)
            cy = int(pt[1] * scale_y)
            cv2.circle(p2, (cx, cy), 3, (0, 100, 255), -1, cv2.LINE_AA)

    short_name = coarse_name[-20:] if coarse_name else "—"
    _text(p2, [f"sim={coarse_sim:.3f}", short_name])
    cv2.putText(p2, "COARSE", (4, _PANEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Panel 3: particle scatter (ENU space)
    # ------------------------------------------------------------------ #
    p3 = _blank()
    if pf.particles is not None and len(pf.particles) > 0:
        pts_e = pf.particles[:, 0]
        pts_n = pf.particles[:, 1]

        pad = max(pf.weighted_spread() * 3.0, 30.0)
        est_e, est_n, _ = pf.estimate()
        min_e, max_e = est_e - pad, est_e + pad
        min_n, max_n = est_n - pad, est_n + pad

        def to_px(e, n):
            px = int((e - min_e) / (max_e - min_e + 1e-9) * (_PANEL_W - 1))
            py = int((1.0 - (n - min_n) / (max_n - min_n + 1e-9)) * (_PANEL_H - 1))
            return np.clip(px, 0, _PANEL_W - 1), np.clip(py, 0, _PANEL_H - 1)

        # Draw grid lines
        for frac in [0.25, 0.5, 0.75]:
            gx = int(frac * _PANEL_W)
            gy = int(frac * _PANEL_H)
            cv2.line(p3, (gx, 0), (gx, _PANEL_H), (40, 40, 40), 1)
            cv2.line(p3, (0, gy), (_PANEL_W, gy), (40, 40, 40), 1)

        # Normalize weights for alpha
        w = pf.weights
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-9)

        # Draw particles colored by weight (blue=low, red=high)
        for i in range(len(pts_e)):
            px, py = to_px(pts_e[i], pts_n[i])
            intensity = int(w_norm[i] * 255)
            color = (255 - intensity, 0, intensity)  # blue -> red
            cv2.circle(p3, (px, py), 2, color, -1)

        # Draw estimate (white cross)
        ex, ey = to_px(est_e, est_n)
        cv2.drawMarker(p3, (ex, ey), (255, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

        # Draw ground truth if available (yellow circle)
        if gt_east is not None and gt_north is not None:
            gx, gy = to_px(gt_east, gt_north)
            cv2.circle(p3, (gx, gy), 5, (0, 255, 255), 2, cv2.LINE_AA)

        # Scale bar: 10m
        bar_m = 10.0
        bar_px = int(bar_m / (max_e - min_e + 1e-9) * _PANEL_W)
        bar_px = max(bar_px, 2)
        cv2.line(p3, (10, _PANEL_H - 10), (10 + bar_px, _PANEL_H - 10), (180, 180, 180), 2)
        _text(p3, [f"10m"], 10, _PANEL_H - 22, scale=0.4, color=(180, 180, 180))

        spread = pf.weighted_spread()
        ess = pf.effective_sample_size()
        _text(p3, [f"ESS={ess:.0f}  sprd={spread:.1f}m"], y0=20)
    else:
        _text(p3, ["no particles"])

    cv2.putText(p3, "PARTICLES", (4, _PANEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Status bar
    # ------------------------------------------------------------------ #
    n_panels = 3
    total_w = _PANEL_W * n_panels + _GAP * (n_panels - 1)
    status = np.zeros((_STATUS_H, total_w, 3), dtype=np.uint8)
    # Colored left bar by phase
    cv2.rectangle(status, (0, 0), (6, _STATUS_H), phase_color, -1)

    err_str = f"{error_m:.1f}m" if error_m >= 0 else "N/A"
    lines = [
        f"t={elapsed_s:.1f}s  phase={phase.name}  err={err_str}  N={len(pf.particles) if pf.particles is not None else 0}",
    ]
    _text(status, lines, x=14, y0=_STATUS_H // 2 + 4, scale=0.52, color=phase_color)

    # ------------------------------------------------------------------ #
    # Assemble
    # ------------------------------------------------------------------ #
    gap = np.zeros((_PANEL_H, _GAP, 3), dtype=np.uint8)
    top_row = np.hstack([p1, gap, p2, gap, p3])
    frame = np.vstack([top_row, status])
    return frame


class DebugVisualizer:
    """
    Wraps build_debug_frame and handles display/publish logic.

    Usage:
        viz = DebugVisualizer(show_window=True)
        ...
        viz.update(pf, drone_frame, ...)
    """

    def __init__(
        self,
        show_window: bool = True,
        window_name: str = "PF Geo-Loc Debug",
        save_frames: bool = False,
        save_dir: Optional[str] = None,
        ros_publisher=None,   # rclpy publisher for sensor_msgs/Image
        cv_bridge=None,
    ):
        self.show_window = show_window
        self.window_name = window_name
        self.save_frames = save_frames
        self.save_dir = save_dir
        self.ros_pub = ros_publisher
        self.bridge = cv_bridge

        self._frame_idx = 0

        # Cached match results (set from outside between calls)
        self.coarse_patch: Optional[np.ndarray] = None
        self.coarse_name: str = ""
        self.coarse_sim: float = 0.0
        self.mkpts_drone: Optional[np.ndarray] = None
        self.mkpts_patch: Optional[np.ndarray] = None
        self.fine_method: str = ""
        self.fine_inliers: int = 0

        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 320 * 3 + 8, 320 + 48 + 4)

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
        """Build and display/publish the debug frame. Returns the image."""
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
