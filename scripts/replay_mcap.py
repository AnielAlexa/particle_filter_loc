#!/usr/bin/env python3
"""Offline MCAP replay for particle filter geo-localization evaluation."""

import argparse
import copy
import csv
import os
import pickle
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# Add package to path
PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from particle_filter_loc.geo_utils import ENUFrame, haversine_m
from particle_filter_loc.motion_model import RTKMotionModel, MotionDelta
from particle_filter_loc.particle_filter import PFConfig, ParticleFilter, Phase
from particle_filter_loc.observation_model import ObservationModel
from particle_filter_loc.debug_viz import DebugVisualizer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_pf_config(pf_cfg_dict: dict) -> PFConfig:
    return PFConfig(**pf_cfg_dict)


def run_replay(
    config,                           # str path or dict
    show_window: bool = False,
    save_frames: bool = False,
    save_cache: bool = False,
    cache_path: Optional[str] = None, # if set, skip TRT and replay from cache
    altitude_min_override: Optional[float] = None,
    max_frames: Optional[int] = None, # stop after this many camera frames (debug)
    bag_name: Optional[str] = None,   # override bag name for output dirs
) -> dict:
    """Run replay and return summary metrics dict.

    Modes:
      - Normal (no cache flags): runs TRT matchers, outputs CSV/plot
      - save_cache=True: same as normal but also writes match_cache.pkl
      - cache_path set: skips MCAP+TRT, replays PF from cache file
    """
    if isinstance(config, str):
        cfg = load_config(config)
    else:
        cfg = copy.deepcopy(config)

    # ENU frame
    enu = ENUFrame(cfg["enu_origin"]["lat"], cfg["enu_origin"]["lon"])

    # PF config
    pf_cfg_dict = cfg["particle_filter"]
    pf_config = _build_pf_config(pf_cfg_dict)

    # Replay config
    rcfg = cfg["replay"]
    mcap_path = rcfg["mcap_path"]
    camera_topic = rcfg.get("camera_topic", "/camera/image_mono")
    rtk_topic    = rcfg.get("rtk_topic",    "/m300/rtk/fix")
    yaw_topic    = rcfg.get("yaw_topic",    "/m300/rtk/yaw")
    altimeter_topic = rcfg.get("altimeter_topic", "/altimeter/range")
    camera_subsample = rcfg.get("camera_subsample", 1)
    start_offset_s   = rcfg.get("start_offset_s", 0.0)
    altitude_min_m   = altitude_min_override if altitude_min_override is not None \
                       else rcfg.get("altitude_min_process_m", 0.0)

    # Derive bag name for outputs
    if bag_name is None:
        bag_stem = Path(mcap_path).stem  # e.g. "Day2.6_0"
        bag_name = bag_stem.replace("_0", "")  # e.g. "Day2.6"

    output_csv  = Path(PKG_DIR) / "results" / bag_name / "pf_results.csv"
    output_plot = Path(PKG_DIR) / "results" / bag_name / "pf_trajectory.png"
    cache_out   = Path(PKG_DIR) / "results" / bag_name / "match_cache.pkl"

    # Legacy single-bag outputs (symlink-style fallback kept for backward compat)
    if "output_csv" in rcfg:
        legacy_csv = Path(PKG_DIR) / rcfg["output_csv"]
        output_csv = legacy_csv
        output_plot = Path(PKG_DIR) / rcfg["output_plot"]
        cache_out = output_csv.parent / "match_cache.pkl"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- CACHE READ MODE ----
    if cache_path is not None:
        return _replay_from_cache(
            cache_path=cache_path,
            pf_config=pf_config,
            lock_cfg=cfg.get("init", {}),
            altitude_min_m=altitude_min_m,
            output_csv=str(output_csv),
            output_plot=str(output_plot),
        )

    # ---- FULL MCAP REPLAY ----
    # Components
    pf = ParticleFilter(pf_config)
    motion = RTKMotionModel()
    matchers_cfg = cfg["matchers"]
    obs = ObservationModel(matchers_cfg, enu)

    # Debug visualizer
    debug_frames_dir = str(output_csv.parent / "debug_frames") if save_frames else None
    viz = DebugVisualizer(
        show_window=show_window,
        save_frames=save_frames,
        save_dir=debug_frames_dir,
    )

    # Coarse-lock parameters (pre-seed accumulator)
    lock_n_frames   = cfg.get("init", {}).get("lock_n_frames", 4)
    lock_sim_thresh = cfg.get("init", {}).get("lock_sim_threshold", 0.40)

    # State
    altitude_buf = deque(maxlen=10)
    current_altitude_m = 0.0
    gt_lat, gt_lon = None, None
    camera_frame_idx = 0
    initialized = False
    results = []
    t_start = None
    tracking_start_ts = None
    fine_attempted = 0
    fine_succeeded = 0

    # Pre-seed coarse-lock state
    _lock_patch: str = ""
    _lock_count: int = 0
    _lock_sims:  list = []

    # Cache accumulation state
    cache_frames = []                 # list of frame dicts (index 0 = header, set at end)
    _pending_rtk_deltas = []          # RTK deltas since last camera frame

    print(f"[{bag_name}] Opening MCAP: {mcap_path}")
    print(f"[{bag_name}] Altitude gating: process frames >= {altitude_min_m:.0f}m")

    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    bag_file = open(mcap_path, "rb")
    reader = make_reader(bag_file, decoder_factories=[DecoderFactory()])

    topics = [camera_topic, rtk_topic, yaw_topic, altimeter_topic]

    print(f"[{bag_name}] Starting replay...")

    for schema, channel, message, decoded_msg in reader.iter_decoded_messages(topics=topics):
        topic = channel.topic
        ts_ns = message.log_time

        if t_start is None:
            t_start = ts_ns

        elapsed_s = (ts_ns - t_start) * 1e-9

        # Skip messages before start offset
        if elapsed_s < start_offset_s:
            continue

        # --- Altimeter ---
        if topic == altimeter_topic:
            alt = float(decoded_msg.range)
            altitude_buf.append(alt)
            current_altitude_m = float(np.median(altitude_buf)) if altitude_buf else alt
            if not initialized and len(altitude_buf) >= 5:
                if pf.try_init(current_altitude_m):
                    initialized = True
                    print(f"  [{bag_name}] Altitude init: median={current_altitude_m:.1f}m "
                          f"> {pf_config.init_altitude_m}m")
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
                    _pending_rtk_deltas.append({
                        "dx_m": delta.dx_m,
                        "dy_m": delta.dy_m,
                        "heading_deg": delta.heading_deg,
                        "dt_s": delta.dt_s,
                    })
            else:
                motion.update(ts_ns, lat=lat, lon=lon)
            continue

        # --- Camera ---
        if topic == camera_topic:
            if not initialized:
                continue

            camera_frame_idx += 1
            if camera_frame_idx % camera_subsample != 0:
                continue

            if max_frames is not None and camera_frame_idx > max_frames:
                print(f"  [{bag_name}] Reached max_frames={max_frames}, stopping.")
                break

            # --- Altitude gating ---
            if altitude_min_m > 0.0 and current_altitude_m < altitude_min_m:
                if save_cache:
                    cache_frames.append({
                        "timestamp_ns": ts_ns,
                        "elapsed_s": elapsed_s,
                        "frame_idx": camera_frame_idx,
                        "altitude_m": current_altitude_m,
                        "gt_lat": gt_lat or 0.0,
                        "gt_lon": gt_lon or 0.0,
                        "gated_out": True,
                        "rtk_deltas": _pending_rtk_deltas,
                    })
                _pending_rtk_deltas = []
                continue

            # Decode image
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
                _pending_rtk_deltas = []
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
                    _lock_patch = top1_name
                    _lock_count = 1
                    _lock_sims  = [top1_sim] if top1_sim >= lock_sim_thresh else []

                print(f"  [{bag_name}|UNINIT] top1={top1_name} sim={top1_sim:.3f}  "
                      f"lock={_lock_count}/{lock_n_frames}")

                # Update viz during UNINIT so the window isn't blank
                viz.coarse_name = top1_name
                viz.coarse_sim = top1_sim
                viz.coarse_top_k_names = coarse.top_k_names
                viz.coarse_top_k_sims = coarse.top_k_sims
                viz.coarse_top_k_patches = [
                    cv2.imread(str(obs.patches_dir / (n + ".png")))
                    for n in coarse.top_k_names
                ]
                viz.coarse_patch = viz.coarse_top_k_patches[0] if viz.coarse_top_k_patches else None
                gt_e = enu.wgs84_to_enu(gt_lat, gt_lon)[0] if gt_lat is not None else None
                gt_n = enu.wgs84_to_enu(gt_lat, gt_lon)[1] if gt_lat is not None else None
                elapsed_s_frame = (ts_ns - t_start) * 1e-9
                viz.update(pf, frame_bgr, error_m=-1.0, elapsed_s=elapsed_s_frame,
                           gt_east=gt_e, gt_north=gt_n)

                if _lock_count >= lock_n_frames and len(_lock_sims) >= lock_n_frames:
                    mean_sim = float(np.mean(_lock_sims))
                    tight_sigma = pf_config.sigma_obs_coarse / np.sqrt(_lock_count)
                    centers_enu = [obs.get_patch_center_enu(n) for n in coarse.top_k_names]
                    pf.seed_from_coarse(centers_enu, coarse.top_k_sims, sigma_override=tight_sigma)
                    print(f"  [{bag_name}] PF seeded after {_lock_count}-frame lock: "
                          f"patch={_lock_patch}  mean_sim={mean_sim:.3f}  "
                          f"seed_sigma={tight_sigma:.1f}m")
                _pending_rtk_deltas = []
                continue

            # --- Coarse match (adaptive radius) ---
            est_e, est_n, est_hdg = pf.estimate()
            search_radius = pf.get_search_radius()

            if pf.phase == Phase.DISPERSED:
                candidate_indices = None
            else:
                candidate_indices = obs.get_indices_within_radius(est_e, est_n, search_radius)
                if len(candidate_indices) < 5:
                    candidate_indices = obs.get_indices_within_radius(est_e, est_n, search_radius * 2.0)

            coarse = obs.coarse_match(frame_bgr, candidate_indices=candidate_indices,
                                       top_k=pf_config.top_k_coarse)

            # Visualizer cache
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

            # Build coarse obs with ENU centers
            coarse_obs = []
            coarse_top_k_enu = []
            for name, sim in zip(coarse.top_k_names, coarse.top_k_sims):
                e, n = obs.get_patch_center_enu(name)
                coarse_obs.append((e, n, sim))
                coarse_top_k_enu.append((e, n))

            pf.update_coarse(coarse_obs, altitude_m=current_altitude_m)

            # Strong trust: if top-1 sim is high, teleport particles there
            if coarse_obs:
                top_e, top_n, top_sim = coarse_obs[0]
                pf.inject_coarse_trust(top_e, top_n, top_sim)

            # --- Fine match (adaptive) ---
            fine_result = None
            viz.mkpts_drone = None
            viz.mkpts_patch = None
            if pf.should_run_fine():
                fine_top_k = pf.get_fine_top_k()
                ctx_frac = pf.get_context_fraction()
                candidates_to_try = coarse.top_k_names[:fine_top_k]
                fine_attempted += 1

                for cand_name in candidates_to_try:
                    fine_result = obs.fine_match(frame_bgr, cand_name, context_fraction=ctx_frac)
                    if fine_result is not None:
                        viz.fine_matched_name = cand_name
                        break

                if fine_result is not None:
                    fine_succeeded += 1
                    fe, fn = enu.wgs84_to_enu(fine_result.lat, fine_result.lon)
                    pf.update_fine(fe, fn, fine_result.inliers, fine_result.heading_deg)
                    viz.fine_method = fine_result.method
                    viz.fine_inliers = fine_result.inliers
                    viz.mkpts_drone = fine_result.mkpts_drone
                    viz.mkpts_patch = fine_result.mkpts_patch
                    viz.fine_H = fine_result.H
                else:
                    viz.mkpts_drone = None
                    viz.mkpts_patch = None
                    viz.fine_H = None

            # Resample + transitions
            pf.resample_if_needed()
            pf.check_transitions()

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
            elapsed_s_frame = (ts_ns - t_start) * 1e-9

            # Debug visualization
            gt_e = enu.wgs84_to_enu(gt_lat, gt_lon)[0] if gt_lat is not None else None
            gt_n = enu.wgs84_to_enu(gt_lat, gt_lon)[1] if gt_lat is not None else None
            viz.update(pf, frame_bgr, error_m=error_m, elapsed_s=elapsed_s_frame,
                       gt_east=gt_e, gt_north=gt_n)

            results.append({
                "timestamp_ns": ts_ns,
                "est_lat": est_lat,
                "est_lon": est_lon,
                "est_heading": est_hdg,
                "gt_lat": gt_lat if gt_lat else 0.0,
                "gt_lon": gt_lon if gt_lon else 0.0,
                "error_m": error_m,
                "altitude_m": current_altitude_m,
                "ess": ess,
                "spread_m": spread,
                "state": pf.phase.name,
                "fine_method": fine_result.method if fine_result else "",
                "fine_inliers": fine_result.inliers if fine_result else 0,
            })

            # Build cache frame
            if save_cache:
                fine_cache = None
                if fine_result is not None:
                    fe_c, fn_c = enu.wgs84_to_enu(fine_result.lat, fine_result.lon)
                    fine_cache = {
                        "east_m": fe_c,
                        "north_m": fn_c,
                        "inliers": fine_result.inliers,
                        "heading_deg": fine_result.heading_deg,
                        "method": fine_result.method,
                        "patch_name": fine_result.patch_name,
                    }
                cache_frames.append({
                    "timestamp_ns": ts_ns,
                    "elapsed_s": elapsed_s_frame,
                    "frame_idx": camera_frame_idx,
                    "altitude_m": current_altitude_m,
                    "gt_lat": gt_lat or 0.0,
                    "gt_lon": gt_lon or 0.0,
                    "coarse_top_k_names": list(coarse.top_k_names),
                    "coarse_top_k_sims":  list(coarse.top_k_sims),
                    "coarse_top_k_enu":   coarse_top_k_enu,
                    "fine_result": fine_cache,
                    "rtk_deltas": _pending_rtk_deltas,
                })

            _pending_rtk_deltas = []

            if camera_frame_idx % 10 == 0:
                fine_rate = f"{fine_succeeded}/{fine_attempted}" if fine_attempted else "0/0"
                print(f"  [{bag_name}|{elapsed_s_frame:6.1f}s] frame={camera_frame_idx:4d}  "
                      f"phase={pf.phase.name:11s}  alt={current_altitude_m:5.1f}m  "
                      f"error={error_m:6.1f}m  "
                      f"spread={spread:5.1f}m  ESS={ess:5.1f}  "
                      f"fine={fine_rate}  t={t_frame_ms:5.1f}ms")

                top1 = coarse.top_k_names[0] if coarse.top_k_names else ""
                if top1 and gt_lat is not None:
                    p = obs.gps_metadata.get(top1, {})
                    p_lat = p.get("lat", 0.0)
                    p_lon = p.get("lon", 0.0)
                    dlat_m = (gt_lat - p_lat) * 111320
                    dlon_m = (gt_lon - p_lon) * 111320 * 0.7071
                    print(f"    patch {top1:12s}  ({p_lat:.6f}, {p_lon:.6f})")
                    print(f"    RTK               ({gt_lat:.6f}, {gt_lon:.6f})")
                    print(f"    delta             N={dlat_m:+.1f}m  E={dlon_m:+.1f}m")

    bag_file.close()
    viz.close()

    # ---- Write cache ----
    if save_cache and cache_frames:
        header = {
            "__cache_version__": 1,
            "bag_name": bag_name,
            "mcap_path": mcap_path,
            "start_offset_s": start_offset_s,
            "altitude_min_process_m": altitude_min_m,
            "enu_origin": {"lat": cfg["enu_origin"]["lat"], "lon": cfg["enu_origin"]["lon"]},
            "pf_config_snapshot": dict(pf_cfg_dict),
            "matchers_config_snapshot": dict(matchers_cfg),
            "n_frames": len(cache_frames),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = [header] + cache_frames
        cache_out.parent.mkdir(parents=True, exist_ok=True)
        with open(str(cache_out), "wb") as f:
            pickle.dump(payload, f, protocol=4)
        print(f"[{bag_name}] Match cache saved: {cache_out}  ({len(cache_frames)} frames)")

    return _finalize_results(results, output_csv, output_plot, t_start,
                              tracking_start_ts, fine_attempted, fine_succeeded, bag_name)


def _replay_from_cache(
    cache_path: str,
    pf_config: PFConfig,
    lock_cfg: dict,
    altitude_min_m: float,
    output_csv: str,
    output_plot: str,
) -> dict:
    """PF-only replay from a match cache file. No TRT, no MCAP."""
    print(f"[cache] Loading: {cache_path}")
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    header = payload[0]
    cache_frames = payload[1:]
    bag_name = header.get("bag_name", Path(cache_path).parent.name)
    enu = ENUFrame(header["enu_origin"]["lat"], header["enu_origin"]["lon"])

    print(f"[{bag_name}|cache] {len(cache_frames)} frames  "
          f"(created {header.get('created_at', '?')})")

    # Warn if matchers config has changed (cache may be stale)
    # (caller can check; we just print)

    lock_n_frames   = lock_cfg.get("lock_n_frames", 4)
    lock_sim_thresh = lock_cfg.get("lock_sim_threshold", 0.40)

    pf = ParticleFilter(pf_config, rng_seed=42)
    results = []
    initialized = False
    tracking_start_ts = None
    fine_attempted = fine_succeeded = 0
    _lock_patch = ""
    _lock_count = 0
    _lock_sims  = []
    t_start_ns  = cache_frames[0]["timestamp_ns"] if cache_frames else 0

    for frame in cache_frames:
        ts_ns   = frame["timestamp_ns"]
        alt     = frame["altitude_m"]
        gt_lat  = frame["gt_lat"]
        gt_lon  = frame["gt_lon"]
        elapsed = frame.get("elapsed_s", (ts_ns - t_start_ns) * 1e-9)

        # Apply RTK deltas (even for gated frames — drone moved)
        for d in frame.get("rtk_deltas", []):
            if pf.particles is not None:
                pf.predict(MotionDelta(**d))

        if frame.get("gated_out"):
            continue

        # Altitude gating (may differ from cache's original gating if config changed)
        if altitude_min_m > 0.0 and alt < altitude_min_m:
            continue

        if not initialized:
            if pf.try_init(alt):
                initialized = True
                print(f"  [{bag_name}|cache] Altitude init: {alt:.1f}m")
            else:
                continue

        coarse_names = frame.get("coarse_top_k_names", [])
        coarse_sims  = frame.get("coarse_top_k_sims",  [])
        coarse_enu   = frame.get("coarse_top_k_enu",   [])

        # Pre-seed lock logic
        if pf.phase == Phase.UNINIT:
            top1_name = coarse_names[0] if coarse_names else ""
            top1_sim  = coarse_sims[0]  if coarse_sims  else 0.0
            if top1_name == _lock_patch and top1_sim >= lock_sim_thresh:
                _lock_count += 1
                _lock_sims.append(top1_sim)
            else:
                _lock_patch = top1_name
                _lock_count = 1
                _lock_sims  = [top1_sim] if top1_sim >= lock_sim_thresh else []

            if _lock_count >= lock_n_frames and len(_lock_sims) >= lock_n_frames:
                tight_sigma = pf_config.sigma_obs_coarse / np.sqrt(_lock_count)
                centers_enu = [c for c in coarse_enu]
                pf.seed_from_coarse(centers_enu, coarse_sims, sigma_override=tight_sigma)
                print(f"  [{bag_name}|cache] PF seeded: sigma={tight_sigma:.1f}m")
            continue

        # Coarse update
        coarse_obs = [(e, n, s) for (e, n), s in zip(coarse_enu, coarse_sims)]
        if coarse_obs:
            pf.update_coarse(coarse_obs, altitude_m=alt)

        # Fine update (use cache's fine result if PF decides to run fine this frame)
        fine_cache = frame.get("fine_result")
        run_fine = pf.should_run_fine()
        if fine_cache and run_fine:
            fine_attempted += 1
            fe = fine_cache["east_m"]
            fn = fine_cache["north_m"]
            pf.update_fine(fe, fn, fine_cache["inliers"], fine_cache.get("heading_deg"))
            fine_succeeded += 1
        elif run_fine:
            fine_attempted += 1

        pf.resample_if_needed()
        pf.check_transitions()

        if pf.phase == Phase.TRACKING and tracking_start_ts is None:
            tracking_start_ts = ts_ns

        est_e, est_n, est_hdg = pf.estimate()
        est_lat, est_lon = enu.enu_to_wgs84(est_e, est_n)
        error_m = haversine_m(est_lat, est_lon, gt_lat, gt_lon) if gt_lat else -1.0
        spread = pf.weighted_spread()
        ess = pf.effective_sample_size()

        results.append({
            "timestamp_ns": ts_ns,
            "est_lat": est_lat,
            "est_lon": est_lon,
            "est_heading": est_hdg,
            "gt_lat": gt_lat,
            "gt_lon": gt_lon,
            "error_m": error_m,
            "altitude_m": alt,
            "ess": ess,
            "spread_m": spread,
            "state": pf.phase.name,
            "fine_method": "",
            "fine_inliers": fine_cache.get("inliers", 0) if fine_cache else 0,
        })

    return _finalize_results(results, output_csv, output_plot, t_start_ns,
                              tracking_start_ts, fine_attempted, fine_succeeded, bag_name)


def _finalize_results(results, output_csv, output_plot, t_start,
                      tracking_start_ts, fine_attempted, fine_succeeded, bag_name) -> dict:
    if not results:
        print(f"[{bag_name}] No results collected.")
        return {"bag_name": bag_name, "n_frames": 0,
                "median_err_all": -1, "median_err_tracking": -1,
                "p90_err": -1, "converge_s": -1, "fine_rate_pct": 0.0}

    # Write CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(str(output_csv), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[{bag_name}] Results saved: {output_csv}")

    # Summary metrics
    errors = [r["error_m"] for r in results if r["error_m"] >= 0]
    tracking_errors = [r["error_m"] for r in results if r["error_m"] >= 0 and r["state"] == "TRACKING"]
    tracking_errors_arr = np.array(tracking_errors) if tracking_errors else np.array([])

    median_all = float(np.median(errors)) if errors else -1.0
    mean_all   = float(np.mean(errors)) if errors else -1.0
    p90_all    = float(np.percentile(errors, 90)) if errors else -1.0
    max_all    = float(np.max(errors)) if errors else -1.0

    median_track = float(np.median(tracking_errors_arr)) if len(tracking_errors_arr) else -1.0
    mean_track   = float(np.mean(tracking_errors_arr)) if len(tracking_errors_arr) else -1.0
    p90_track    = float(np.percentile(tracking_errors_arr, 90)) if len(tracking_errors_arr) else -1.0

    converge_s = -1.0
    if tracking_start_ts and t_start:
        converge_s = float((tracking_start_ts - t_start) * 1e-9)

    fine_rate_pct = 100.0 * fine_succeeded / max(fine_attempted, 1)

    print(f"\n[{bag_name}] === Summary ===")
    if errors:
        errors_np = np.array(errors)
        print(f"  All frames:      median={median_all:.1f}m  "
              f"mean={mean_all:.1f}m  "
              f"90th={p90_all:.1f}m  "
              f"max={max_all:.1f}m")
    if len(tracking_errors_arr):
        print(f"  TRACKING only:   median={median_track:.1f}m  "
              f"mean={mean_track:.1f}m  "
              f"90th={p90_track:.1f}m")
    if converge_s >= 0:
        print(f"  Time to TRACKING: {converge_s:.1f}s")
    print(f"  Total frames: {len(results)}")
    print(f"  Fine match rate: {fine_succeeded}/{fine_attempted} ({fine_rate_pct:.1f}%)")

    # Plot
    try:
        _generate_plot(results, str(output_plot), t_start, bag_name)
        print(f"[{bag_name}] Plot saved: {output_plot}")
    except Exception as e:
        print(f"[{bag_name}] Plot failed: {e}")

    return {
        "bag_name": bag_name,
        "n_frames": len(results),
        "median_err_all": median_all,
        "mean_err_all": mean_all,
        "p90_err": p90_all,
        "max_err": max_all,
        "median_err_tracking": median_track,
        "mean_err_tracking": mean_track,
        "p90_err_tracking": p90_track,
        "converge_s": converge_s,
        "fine_rate_pct": fine_rate_pct,
        "fine_succeeded": fine_succeeded,
        "fine_attempted": fine_attempted,
    }


def _generate_plot(results: list, output_path: str, t_start: int, bag_name: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.array([(r["timestamp_ns"] - t_start) * 1e-9 for r in results])
    est_lat = np.array([r["est_lat"] for r in results])
    est_lon = np.array([r["est_lon"] for r in results])
    gt_lat  = np.array([r["gt_lat"]  for r in results])
    gt_lon  = np.array([r["gt_lon"]  for r in results])
    errors  = np.array([r["error_m"] for r in results])
    ess     = np.array([r["ess"]     for r in results])
    spread  = np.array([r["spread_m"] for r in results])
    alts    = np.array([r.get("altitude_m", 0.0) for r in results])
    states  = [r["state"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"PF Evaluation — {bag_name}", fontsize=13)

    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(gt_lon, gt_lat, "b-", alpha=0.5, label="GT (RTK)", linewidth=1)
    ax.plot(est_lon, est_lat, "r-", alpha=0.7, label="PF estimate", linewidth=1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Trajectory")
    ax.legend()
    ax.set_aspect("equal")

    # 2. Error vs time
    ax = axes[0, 1]
    tracking_mask = np.array([s == "TRACKING" for s in states])
    ax.plot(ts, errors, "k-", linewidth=0.8, alpha=0.5, label="all")
    if tracking_mask.any():
        ax.plot(ts[tracking_mask], errors[tracking_mask], "g-",
                linewidth=1.2, label="TRACKING")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Localization Error")
    ax.set_ylim(bottom=0)
    ax.legend()

    # 3. Altitude vs time
    ax = axes[0, 2]
    ax.plot(ts, alts, "c-", linewidth=0.8)
    ax.axhline(50.0, color="orange", linestyle="--", linewidth=0.8, label="50m gate")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude")
    ax.legend()

    # 4. ESS and spread
    ax = axes[1, 0]
    ax.plot(ts, ess, "g-", linewidth=0.8, label="ESS")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESS", color="g")
    ax2 = ax.twinx()
    ax2.plot(ts, spread, "m-", linewidth=0.8, label="Spread (m)")
    ax2.set_ylabel("Spread (m)", color="m")
    ax.set_title("ESS & Spread")

    # 5. State timeline
    ax = axes[1, 1]
    state_map = {"UNINIT": 0, "DISPERSED": 1, "CONVERGING": 2, "TRACKING": 3}
    state_vals = [state_map.get(s, 0) for s in states]
    ax.step(ts, state_vals, "b-", linewidth=1.5, where="post")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["UNINIT", "DISPERSED", "CONVERGING", "TRACKING"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Phase Timeline")

    # 6. Error CDF (tracking only)
    ax = axes[1, 2]
    track_errs = errors[tracking_mask]
    if len(track_errs) > 0:
        sorted_e = np.sort(track_errs)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax.plot(sorted_e, cdf * 100, "g-", linewidth=1.5)
        ax.axvline(float(np.median(sorted_e)), color="r", linestyle="--",
                   linewidth=0.8, label=f"median={np.median(sorted_e):.1f}m")
        ax.axvline(float(np.percentile(sorted_e, 90)), color="orange", linestyle="--",
                   linewidth=0.8, label=f"p90={np.percentile(sorted_e, 90):.1f}m")
        ax.legend(fontsize=8)
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Error CDF (TRACKING)")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--save-cache", action="store_true",
                        help="Save match cache to results/<bag_name>/match_cache.pkl")
    parser.add_argument("--use-cache", metavar="PATH",
                        help="Replay PF from match cache (no TRT, no MCAP)")
    parser.add_argument("--altitude-min", type=float, default=None,
                        help="Override altitude_min_process_m from config")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N camera frames (debug)")
    args = parser.parse_args()
    run_replay(
        args.config,
        show_window=args.show,
        save_frames=args.save_frames,
        save_cache=args.save_cache,
        cache_path=args.use_cache,
        altitude_min_override=args.altitude_min,
        max_frames=args.max_frames,
    )
