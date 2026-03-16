#!/usr/bin/env python3
"""Offline MCAP replay for particle filter geo-localization evaluation."""

import argparse
import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add package to path
PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from particle_filter_loc.geo_utils import ENUFrame, haversine_m
from particle_filter_loc.motion_model import RTKMotionModel
from particle_filter_loc.particle_filter import PFConfig, ParticleFilter, Phase
from particle_filter_loc.observation_model import ObservationModel
from particle_filter_loc.debug_viz import DebugVisualizer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_replay(config_path: str, show_window: bool = False, save_frames: bool = False):
    cfg = load_config(config_path)

    # ENU frame
    enu = ENUFrame(cfg["enu_origin"]["lat"], cfg["enu_origin"]["lon"])

    # PF config
    pf_cfg_dict = cfg["particle_filter"]
    pf_config = PFConfig(**pf_cfg_dict)

    # Components
    pf = ParticleFilter(pf_config)
    motion = RTKMotionModel()
    obs = ObservationModel(cfg["matchers"], enu)

    # Replay config
    rcfg = cfg["replay"]
    mcap_path = rcfg["mcap_path"]
    camera_topic = rcfg["camera_topic"]
    rtk_topic = rcfg["rtk_topic"]
    yaw_topic = rcfg["yaw_topic"]
    altimeter_topic = rcfg["altimeter_topic"]
    camera_subsample = rcfg.get("camera_subsample", 1)
    output_csv = Path(PKG_DIR) / rcfg["output_csv"]
    output_plot = Path(PKG_DIR) / rcfg["output_plot"]

    # Ensure output dirs exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    # Open MCAP
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    print(f"Opening MCAP: {mcap_path}")
    bag_file = open(mcap_path, "rb")
    reader = make_reader(bag_file, decoder_factories=[DecoderFactory()])

    topics = [camera_topic, rtk_topic, yaw_topic, altimeter_topic]

    # Debug visualizer
    debug_frames_dir = str(output_csv.parent / "debug_frames") if save_frames else None
    viz = DebugVisualizer(
        show_window=show_window,
        save_frames=save_frames,
        save_dir=debug_frames_dir,
    )

    # Coarse-lock parameters (pre-seed accumulator)
    lock_n_frames    = cfg.get("init", {}).get("lock_n_frames", 4)
    lock_sim_thresh  = cfg.get("init", {}).get("lock_sim_threshold", 0.40)

    # State
    altitude_buf = deque(maxlen=10)
    gt_lat, gt_lon = None, None
    camera_frame_idx = 0
    initialized = False
    results = []
    t_start = None
    tracking_start_ts = None
    fine_attempted = 0
    fine_succeeded = 0

    # Pre-seed coarse-lock state
    _lock_patch: str = ""          # current candidate patch
    _lock_count: int = 0           # consecutive hits on that patch
    _lock_sims:  list = []         # sim scores during the lock streak

    print("Starting replay...")

    for schema, channel, message, decoded_msg in reader.iter_decoded_messages(topics=topics):
        topic = channel.topic
        ts_ns = message.log_time

        if t_start is None:
            t_start = ts_ns

        # --- Altimeter ---
        if topic == altimeter_topic:
            # Range message — .range field
            alt = float(decoded_msg.range)
            altitude_buf.append(alt)
            if not initialized and len(altitude_buf) >= 5:
                median_alt = float(np.median(altitude_buf))
                if pf.try_init(median_alt):
                    initialized = True
                    print(f"  Altitude init: median={median_alt:.1f}m > {pf_config.init_altitude_m}m")
            continue

        # --- Yaw ---
        if topic == yaw_topic:
            motion.set_yaw(float(decoded_msg.data))
            continue

        # --- RTK ---
        if topic == rtk_topic:
            lat = decoded_msg.latitude
            lon = decoded_msg.longitude
            gt_lat, gt_lon = lat, lon

            if initialized and pf.phase != Phase.UNINIT:
                delta = motion.update(ts_ns, lat=lat, lon=lon)
                if delta is not None:
                    pf.predict(delta)
            else:
                # Still need to feed RTK to motion model for delta tracking
                motion.update(ts_ns, lat=lat, lon=lon)
            continue

        # --- Camera ---
        if topic == camera_topic:
            if not initialized:
                continue

            camera_frame_idx += 1
            if camera_frame_idx % camera_subsample != 0:
                continue

            # Decode mono8 image
            h = decoded_msg.height
            w = decoded_msg.width
            if decoded_msg.encoding == "mono8":
                frame_gray = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(h, w)
                frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            elif decoded_msg.encoding in ("bgr8", "rgb8"):
                frame_bgr = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(h, w, 3)
                if decoded_msg.encoding == "rgb8":
                    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            else:
                continue

            t_frame_start = time.monotonic()

            # --- Pre-seed: accumulate coarse-lock before committing PF ---
            if pf.phase == Phase.UNINIT:
                coarse = obs.coarse_match(frame_bgr, candidate_indices=None,
                                          top_k=pf_config.top_k_coarse)
                top1_name = coarse.top_k_names[0] if coarse.top_k_names else ""
                top1_sim  = coarse.top_k_sims[0]  if coarse.top_k_sims  else 0.0

                if top1_name == _lock_patch and top1_sim >= lock_sim_thresh:
                    _lock_count += 1
                    _lock_sims.append(top1_sim)
                else:
                    # Reset streak — new candidate
                    _lock_patch = top1_name
                    _lock_count = 1
                    _lock_sims  = [top1_sim] if top1_sim >= lock_sim_thresh else []

                print(f"  [UNINIT] top1={top1_name} sim={top1_sim:.3f}  "
                      f"lock={_lock_count}/{lock_n_frames}")

                if _lock_count >= lock_n_frames and len(_lock_sims) >= lock_n_frames:
                    # Confirmed lock — seed PF with spread tightened by sqrt(N)
                    mean_sim = float(np.mean(_lock_sims))
                    tight_sigma = pf_config.sigma_obs_coarse / np.sqrt(_lock_count)
                    centers_enu = [obs.get_patch_center_enu(n) for n in coarse.top_k_names]
                    # Temporarily override seed spread
                    orig_sigma = pf_config.sigma_obs_coarse
                    pf_config.sigma_obs_coarse = tight_sigma
                    pf.seed_from_coarse(centers_enu, coarse.top_k_sims)
                    pf_config.sigma_obs_coarse = orig_sigma
                    print(f"  PF seeded after {_lock_count}-frame lock: "
                          f"patch={_lock_patch}  mean_sim={mean_sim:.3f}  "
                          f"seed_sigma={tight_sigma:.1f}m")
                continue

            # --- Coarse match (adaptive radius) ---
            est_e, est_n, est_hdg = pf.estimate()
            search_radius = pf.get_search_radius()

            if pf.phase == Phase.DISPERSED:
                # Full DB search when dispersed
                candidate_indices = None
            else:
                candidate_indices = obs.get_indices_within_radius(est_e, est_n, search_radius)
                if len(candidate_indices) < 5:
                    candidate_indices = None  # fall back to full DB

            coarse = obs.coarse_match(frame_bgr, candidate_indices=candidate_indices,
                                       top_k=pf_config.top_k_coarse)

            # Cache coarse result in visualizer
            viz.coarse_name = coarse.top_k_names[0] if coarse.top_k_names else ""
            viz.coarse_sim = coarse.top_k_sims[0] if coarse.top_k_sims else 0.0
            if viz.coarse_name:
                patch_path = str(obs.patches_dir / (viz.coarse_name + ".png"))
                viz.coarse_patch = cv2.imread(patch_path)
            viz.mkpts_drone = None
            viz.mkpts_patch = None
            viz.fine_method = ""
            viz.fine_inliers = 0

            # Update PF with coarse observations
            coarse_obs = []
            for name, sim in zip(coarse.top_k_names, coarse.top_k_sims):
                e, n = obs.get_patch_center_enu(name)
                coarse_obs.append((e, n, sim))
            pf.update_coarse(coarse_obs)

            # --- Fine match (adaptive) ---
            fine_result = None
            if pf.should_run_fine():
                fine_top_k = pf.get_fine_top_k()
                ctx_frac = pf.get_context_fraction()
                candidates_to_try = coarse.top_k_names[:fine_top_k]
                fine_attempted += 1

                for cand_name in candidates_to_try:
                    fine_result = obs.fine_match(frame_bgr, cand_name, context_fraction=ctx_frac)
                    if fine_result is not None:
                        break

                if fine_result is not None:
                    fine_succeeded += 1
                    fe, fn = enu.wgs84_to_enu(fine_result.lat, fine_result.lon)
                    pf.update_fine(fe, fn, fine_result.inliers, fine_result.heading_deg)
                    viz.fine_method = fine_result.method
                    viz.fine_inliers = fine_result.inliers

            # Resample + transitions
            pf.resample_if_needed()
            pf.check_transitions()

            # Track when we first enter TRACKING
            if pf.phase == Phase.TRACKING and tracking_start_ts is None:
                tracking_start_ts = ts_ns

            # Estimate
            est_e, est_n, est_hdg = pf.estimate()
            est_lat, est_lon = enu.enu_to_wgs84(est_e, est_n)

            # Error vs ground truth
            error_m = haversine_m(est_lat, est_lon, gt_lat, gt_lon) if gt_lat is not None else -1.0
            ess = pf.effective_sample_size()
            spread = pf.weighted_spread()

            t_frame_ms = (time.monotonic() - t_frame_start) * 1000
            elapsed_s = (ts_ns - t_start) * 1e-9

            # Debug visualization
            gt_e = enu.wgs84_to_enu(gt_lat, gt_lon)[0] if gt_lat is not None else None
            gt_n = enu.wgs84_to_enu(gt_lat, gt_lon)[1] if gt_lat is not None else None
            viz.update(pf, frame_bgr, error_m=error_m, elapsed_s=elapsed_s,
                       gt_east=gt_e, gt_north=gt_n)

            results.append({
                "timestamp_ns": ts_ns,
                "est_lat": est_lat,
                "est_lon": est_lon,
                "est_heading": est_hdg,
                "gt_lat": gt_lat if gt_lat else 0.0,
                "gt_lon": gt_lon if gt_lon else 0.0,
                "error_m": error_m,
                "ess": ess,
                "spread_m": spread,
                "state": pf.phase.name,
                "fine_method": fine_result.method if fine_result else "",
                "fine_inliers": fine_result.inliers if fine_result else 0,
            })

            if camera_frame_idx % 10 == 0:
                fine_rate = f"{fine_succeeded}/{fine_attempted}" if fine_attempted else "0/0"
                print(f"  [{elapsed_s:6.1f}s] frame={camera_frame_idx:4d}  "
                      f"phase={pf.phase.name:11s}  error={error_m:6.1f}m  "
                      f"spread={spread:5.1f}m  ESS={ess:5.1f}  "
                      f"fine={fine_rate}  t={t_frame_ms:5.1f}ms")

    bag_file.close()
    viz.close()

    if not results:
        print("No results collected.")
        return

    # --- Write CSV ---
    fieldnames = list(results[0].keys())
    with open(str(output_csv), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_csv}")

    # --- Summary metrics ---
    errors = [r["error_m"] for r in results if r["error_m"] >= 0]
    tracking_errors = [r["error_m"] for r in results if r["error_m"] >= 0 and r["state"] == "TRACKING"]

    print("\n=== Summary ===")
    if errors:
        errors_np = np.array(errors)
        print(f"All frames:      median={np.median(errors_np):.1f}m  "
              f"mean={np.mean(errors_np):.1f}m  "
              f"90th={np.percentile(errors_np, 90):.1f}m  "
              f"max={np.max(errors_np):.1f}m")
    if tracking_errors:
        te = np.array(tracking_errors)
        print(f"TRACKING only:   median={np.median(te):.1f}m  "
              f"mean={np.mean(te):.1f}m  "
              f"90th={np.percentile(te, 90):.1f}m  "
              f"max={np.max(te):.1f}m")
    if tracking_start_ts and t_start:
        convergence_s = (tracking_start_ts - t_start) * 1e-9
        print(f"Time to TRACKING: {convergence_s:.1f}s")
    print(f"Total frames processed: {len(results)}")
    print(f"Fine match rate: {fine_succeeded}/{fine_attempted} "
          f"({100*fine_succeeded/max(fine_attempted,1):.1f}%)")

    # --- Plot ---
    try:
        _generate_plot(results, str(output_plot), t_start)
        print(f"Plot saved to {output_plot}")
    except Exception as e:
        print(f"Plot generation failed: {e}")


def _generate_plot(results: list, output_path: str, t_start: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.array([(r["timestamp_ns"] - t_start) * 1e-9 for r in results])
    est_lat = np.array([r["est_lat"] for r in results])
    est_lon = np.array([r["est_lon"] for r in results])
    gt_lat = np.array([r["gt_lat"] for r in results])
    gt_lon = np.array([r["gt_lon"] for r in results])
    errors = np.array([r["error_m"] for r in results])
    ess = np.array([r["ess"] for r in results])
    spread = np.array([r["spread_m"] for r in results])
    states = [r["state"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(gt_lon, gt_lat, "b-", alpha=0.5, label="GT", linewidth=1)
    ax.plot(est_lon, est_lat, "r-", alpha=0.7, label="PF estimate", linewidth=1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Trajectory")
    ax.legend()
    ax.set_aspect("equal")

    # 2. Error vs time
    ax = axes[0, 1]
    ax.plot(ts, errors, "k-", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Localization Error")
    ax.set_ylim(bottom=0)

    # 3. ESS and spread
    ax = axes[1, 0]
    ax.plot(ts, ess, "g-", linewidth=0.8, label="ESS")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS", color="g")
    ax2 = ax.twinx()
    ax2.plot(ts, spread, "m-", linewidth=0.8, label="Spread (m)")
    ax2.set_ylabel("Spread (m)", color="m")
    ax.set_title("ESS & Spread")

    # 4. State timeline
    ax = axes[1, 1]
    state_map = {"UNINIT": 0, "DISPERSED": 1, "CONVERGING": 2, "TRACKING": 3}
    state_vals = [state_map.get(s, 0) for s in states]
    ax.step(ts, state_vals, "b-", linewidth=1.5, where="post")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["UNINIT", "DISPERSED", "CONVERGING", "TRACKING"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Phase Timeline")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Particle Filter MCAP Replay")
    parser.add_argument("--config", default=str(PKG_DIR / "config" / "pf_config.yaml"),
                        help="Path to pf_config.yaml")
    parser.add_argument("--show", action="store_true",
                        help="Show live OpenCV debug window")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save debug frames as JPEGs to results/debug_frames/")
    args = parser.parse_args()
    run_replay(args.config, show_window=args.show, save_frames=args.save_frames)
