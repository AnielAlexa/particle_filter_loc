#!/usr/bin/env python3
"""ROS2 particle filter geo-localization node (skeleton for future live deployment)."""

import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

import rclpy
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix, Image, Range
from std_msgs.msg import Float32, Float64, String
from cv_bridge import CvBridge

from .geo_utils import ENUFrame
from .motion_model import RTKMotionModel
from .particle_filter import PFConfig, ParticleFilter, Phase
from .observation_model import ObservationModel
from .debug_viz import DebugVisualizer


class PFGeoLocNode(rclpy.node.Node):

    def __init__(self):
        super().__init__("pf_geo_loc_node")

        self._processing = False
        self._lock = threading.Lock()

        config_path = self.declare_parameter("config_path", "").get_parameter_value().string_value
        if not config_path:
            config_path = str(Path(__file__).resolve().parent.parent / "config" / "pf_config.yaml")

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.enu = ENUFrame(self.cfg["enu_origin"]["lat"], self.cfg["enu_origin"]["lon"])

        pf_cfg = PFConfig(**self.cfg["particle_filter"])
        self.pf = ParticleFilter(pf_cfg)
        self.motion = RTKMotionModel()
        self.obs = ObservationModel(self.cfg["matchers"], self.enu)
        self.bridge = CvBridge()

        self._altitude_buf = deque(maxlen=10)
        self._initialized = False
        self._first_init_camera = True
        self._gt_lat = None
        self._gt_lon = None
        self._t_start = None

        self._init_ros()
        self.get_logger().info("PFGeoLocNode ready.")

    def _init_ros(self):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        rcfg = self.cfg.get("replay", {})
        self.create_subscription(Image, rcfg.get("camera_topic", "/camera/image_mono"), self._image_cb, qos)
        self.create_subscription(NavSatFix, rcfg.get("rtk_topic", "/m300/rtk/fix"), self._rtk_cb, 10)
        self.create_subscription(Float64, rcfg.get("yaw_topic", "/m300/rtk/yaw"), self._yaw_cb, 10)
        self.create_subscription(Range, rcfg.get("altimeter_topic", "/altimeter/range"), self._alt_cb, 10)

        self._pub_position = self.create_publisher(NavSatFix, "/pf_geo_loc/position", 10)
        self._pub_ess = self.create_publisher(Float32, "/pf_geo_loc/ess", 10)
        self._pub_state = self.create_publisher(String, "/pf_geo_loc/state", 10)
        self._pub_debug = self.create_publisher(Image, "/pf_geo_loc/debug_view", 1)

        self._viz = DebugVisualizer(
            show_window=False,
            ros_publisher=self._pub_debug,
            cv_bridge=self.bridge,
        )

    def _alt_cb(self, msg):
        self._altitude_buf.append(msg.range)
        if not self._initialized and len(self._altitude_buf) >= 5:
            median_alt = float(np.median(self._altitude_buf))
            if self.pf.try_init(median_alt):
                self._initialized = True
                self.get_logger().info(f"Altitude init: {median_alt:.1f}m")

    def _yaw_cb(self, msg):
        self.motion.set_yaw(msg.data)

    def _rtk_cb(self, msg):
        self._gt_lat = msg.latitude
        self._gt_lon = msg.longitude
        ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if self._initialized and self.pf.phase != Phase.UNINIT:
            delta = self.motion.update(ts_ns, lat=msg.latitude, lon=msg.longitude)
            if delta is not None:
                self.pf.predict(delta)
        else:
            self.motion.update(ts_ns, lat=msg.latitude, lon=msg.longitude)

    def _image_cb(self, msg):
        if not self._initialized:
            return
        with self._lock:
            if self._processing:
                return
            self._processing = True

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._process_frame(frame_bgr)
        finally:
            with self._lock:
                self._processing = False

    def _process_frame(self, frame_bgr):
        pf = self.pf
        obs = self.obs
        enu = self.enu
        pf_config = pf.cfg

        # Seed on first frame
        if pf.phase == Phase.UNINIT and self._first_init_camera:
            self._first_init_camera = False
            coarse = obs.coarse_match(frame_bgr, top_k=pf_config.top_k_coarse)
            centers = [obs.get_patch_center_enu(n) for n in coarse.top_k_names]
            pf.seed_from_coarse(centers, coarse.top_k_sims)
            return

        if pf.phase == Phase.UNINIT:
            return

        est_e, est_n, _ = pf.estimate()
        search_radius = pf.get_search_radius()

        if pf.phase == Phase.DISPERSED:
            candidate_indices = None
        else:
            candidate_indices = obs.get_indices_within_radius(est_e, est_n, search_radius)
            # Expand radius up to 2× if too few patches (edge-of-coverage),
            # but never fall back to unconstrained full-database search.
            if len(candidate_indices) < 5:
                candidate_indices = obs.get_indices_within_radius(est_e, est_n, search_radius * 2.0)

        coarse = obs.coarse_match(frame_bgr, candidate_indices=candidate_indices,
                                   top_k=pf_config.top_k_coarse)

        # Cache coarse result for debug viz
        viz = self._viz
        viz.coarse_name = coarse.top_k_names[0] if coarse.top_k_names else ""
        viz.coarse_sim = coarse.top_k_sims[0] if coarse.top_k_sims else 0.0
        viz.coarse_top_k_names = coarse.top_k_names
        viz.coarse_top_k_sims = coarse.top_k_sims
        viz.coarse_top_k_patches = [
            cv2.imread(str(obs.patches_dir / (n + ".png")))
            for n in coarse.top_k_names
        ]
        viz.coarse_patch = viz.coarse_top_k_patches[0] if viz.coarse_top_k_patches else None
        viz.fine_matched_name = ""
        viz.mkpts_drone = None
        viz.mkpts_patch = None
        viz.fine_method = ""
        viz.fine_inliers = 0

        coarse_obs = []
        for name, sim in zip(coarse.top_k_names, coarse.top_k_sims):
            e, n = obs.get_patch_center_enu(name)
            coarse_obs.append((e, n, sim))
        pf.update_coarse(coarse_obs)

        # Update observation model with current altitude before fine matching
        if len(self._altitude_buf) > 0:
            self.obs.altitude_m = float(np.median(self._altitude_buf))

        fine_result = None
        if pf.should_run_fine():
            fine_top_k = pf.get_fine_top_k()
            ctx_frac = pf.get_context_fraction()
            for cand_name in coarse.top_k_names[:fine_top_k]:
                fine_result = obs.fine_match(frame_bgr, cand_name, context_fraction=ctx_frac)
                if fine_result is not None:
                    fe, fn = enu.wgs84_to_enu(fine_result.lat, fine_result.lon)
                    pf.update_fine(fe, fn, fine_result.inliers, fine_result.heading_deg)
                    viz.fine_method = fine_result.method
                    viz.fine_inliers = fine_result.inliers
                    viz.fine_matched_name = cand_name
                    break

        pf.resample_if_needed()
        pf.check_transitions()

        # Estimate & publish
        est_e, est_n, est_hdg = pf.estimate()
        est_lat, est_lon = enu.enu_to_wgs84(est_e, est_n)

        pos_msg = NavSatFix()
        pos_msg.latitude = est_lat
        pos_msg.longitude = est_lon
        self._pub_position.publish(pos_msg)

        ess_msg = Float32()
        ess_msg.data = float(pf.effective_sample_size())
        self._pub_ess.publish(ess_msg)

        state_msg = String()
        state_msg.data = pf.phase.name
        self._pub_state.publish(state_msg)

        # Debug view
        now_ns = self.get_clock().now().nanoseconds
        if self._t_start is None:
            self._t_start = now_ns
        elapsed_s = (now_ns - self._t_start) * 1e-9
        gt_e, gt_n = None, None
        if self._gt_lat is not None:
            gt_e, gt_n = enu.wgs84_to_enu(self._gt_lat, self._gt_lon)
        viz.update(pf, frame_bgr, error_m=-1.0, elapsed_s=elapsed_s,
                   gt_east=gt_e, gt_north=gt_n)


def main(args=None):
    # Jetson Orin: limit CPU threads — pipeline is GPU-dominated,
    # small arrays (300 particles, 320px images) don't benefit from parallel BLAS.
    import torch as _torch
    _torch.set_num_threads(2)
    _torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)

    rclpy.init(args=args)
    node = PFGeoLocNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
