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
from particle_filter_loc.footprint_reconstruction import SatelliteFootprintReconstructor
from particle_filter_loc.trust_model import (
    TrustConfig, TrustTracker, evaluate_coarse_trust, GlobalCorrectionResult,
)


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
    rtk_init: bool = True,            # seed PF from first RTK fix (skip coarse-lock)
    rtk_noise_m: float = 0.0,         # add Gaussian noise to RTK deltas (sigma in meters)
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

    # Trust config
    _trust_keys = {"trust_recon_sim_min", "trust_coarse_sim_min", "trust_high_inliers",
                   "trust_cross_agree_m", "trust_drift_recon_sim", "trust_drift_coarse_sim",
                   "coarse_trust_sim", "coarse_trust_fraction", "coarse_trust_sigma"}
    trust_cfg_dict = cfg.get("trust", {})
    # Handle coarse_teleport_sigma_range as list
    trust_config = TrustConfig(**trust_cfg_dict)
    trust_tracker = TrustTracker(trust_config)

    pf_config = _build_pf_config({k: v for k, v in pf_cfg_dict.items() if k not in _trust_keys})

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
            trust_config=trust_config,
            lock_cfg=cfg.get("init", {}),
            altitude_min_m=altitude_min_m,
            output_csv=str(output_csv),
            output_plot=str(output_plot),
            rtk_noise_m=rtk_noise_m,
        )

    # ---- FULL MCAP REPLAY ----
    # Components
    pf = ParticleFilter(pf_config)
    motion = RTKMotionModel()
    matchers_cfg = cfg["matchers"]
    obs = ObservationModel(matchers_cfg, enu)

    # Satellite footprint reconstructor
    reconstructor = SatelliteFootprintReconstructor(
        obs.gps_metadata, obs.patches_dir, enu,
    )
    cam_orig = cfg.get("camera_original", {})
    cam_fx = cam_orig.get("fx", 1129.0)
    cam_fy = cam_orig.get("fy", 1130.0)
    cam_w  = cam_orig.get("w", 1280)
    cam_h  = cam_orig.get("h", 720)
    heading_offset_deg = cam_orig.get("heading_offset_deg", 0.0)

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

    # Previous PF estimate (for mosaic fine matching)
    est_lat_prev, est_lon_prev, est_hdg_prev = None, None, 0.0

    # Pre-seed coarse-lock state
    _lock_patch: str = ""
    _lock_count: int = 0
    _lock_sims:  list = []

    # Cache accumulation state
    cache_frames = []                 # list of frame dicts (index 0 = header, set at end)
    _pending_rtk_deltas = []          # RTK deltas since last camera frame

    print(f"[{bag_name}] Opening MCAP: {mcap_path}")
    if rtk_noise_m > 0.0:
        print(f"[{bag_name}] RTK noise injected: sigma={rtk_noise_m:.1f}m  heading_sigma={rtk_noise_m*2:.1f}deg")
    print(f"[{bag_name}] Altitude gating: process frames >= {altitude_min_m:.0f}m")

    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    bag_file = open(mcap_path, "rb")
    reader = make_reader(bag_file, decoder_factories=[DecoderFactory()])

    topics = [camera_topic, rtk_topic, yaw_topic, altimeter_topic]

    print(f"[{bag_name}] Starting replay...")
    _dbg_first_msg = True
    _dbg_offset_done = False
    _dbg_alt_init_printed = False
    _dbg_camera_count = 0

    for schema, channel, message, decoded_msg in reader.iter_decoded_messages(topics=topics):
        topic = channel.topic
        ts_ns = message.log_time

        if t_start is None:
            t_start = ts_ns

        elapsed_s = (ts_ns - t_start) * 1e-9

        if _dbg_first_msg:
            print(f"  [DBG] First message at {elapsed_s:.1f}s  topic={topic}")
            _dbg_first_msg = False

        # Skip messages before start offset
        if elapsed_s < start_offset_s:
            continue

        if not _dbg_offset_done:
            print(f"  [DBG] Past start_offset_s={start_offset_s}s at elapsed={elapsed_s:.1f}s")
            _dbg_offset_done = True

        # --- Altimeter ---
        if topic == altimeter_topic:
            alt = float(decoded_msg.range)
            altitude_buf.append(alt)
            current_altitude_m = float(np.median(altitude_buf)) if altitude_buf else alt
            if not initialized and not _dbg_alt_init_printed:
                print(f"  [DBG] Altimeter: alt={alt:.1f}m  median={current_altitude_m:.1f}m  "
                      f"buf_len={len(altitude_buf)}/5  need>{pf_config.init_altitude_m}m")
            if not initialized and len(altitude_buf) >= 5:
                if pf.try_init(current_altitude_m):
                    initialized = True
                    _dbg_alt_init_printed = True
                    print(f"  [{bag_name}] Altitude init: median={current_altitude_m:.1f}m "
                          f"> {pf_config.init_altitude_m}m")
                elif not _dbg_alt_init_printed and len(altitude_buf) >= 5:
                    print(f"  [DBG] Altimeter buf full but median={current_altitude_m:.1f}m "
                          f"< {pf_config.init_altitude_m}m, waiting...")
                    _dbg_alt_init_printed = True  # only print once
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

            # RTK-init: seed PF directly from first RTK fix once altitude is sufficient
            if rtk_init and initialized and pf.phase == Phase.UNINIT:
                rtk_e, rtk_n = enu.wgs84_to_enu(lat, lon)
                pf.seed_from_position(rtk_e, rtk_n, heading_deg=motion._yaw_deg,
                                      sigma_pos=5.0, sigma_hdg=10.0)
                print(f"  [{bag_name}] RTK-init: seeded PF at ({lat:.6f}, {lon:.6f})  "
                      f"alt={current_altitude_m:.1f}m  hdg={motion._yaw_deg:.0f}")

            if initialized and pf.phase != Phase.UNINIT:
                delta = motion.update(ts_ns, lat=lat, lon=lon)
                if delta is not None:
                    # Inject noise to simulate GPS degradation
                    if rtk_noise_m > 0.0:
                        delta = MotionDelta(
                            dx_m=delta.dx_m + np.random.normal(0, rtk_noise_m),
                            dy_m=delta.dy_m + np.random.normal(0, rtk_noise_m),
                            heading_deg=delta.heading_deg + np.random.normal(0, rtk_noise_m * 2.0),
                            dt_s=delta.dt_s,
                        )
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
            _dbg_camera_count += 1
            if not initialized:
                if _dbg_camera_count <= 3:
                    print(f"  [DBG] Camera frame #{_dbg_camera_count} at {elapsed_s:.1f}s — skipped (not initialized)")
                continue

            camera_frame_idx += 1
            if camera_frame_idx % camera_subsample != 0:
                continue

            if max_frames is not None and camera_frame_idx > max_frames:
                print(f"  [{bag_name}] Reached max_frames={max_frames}, stopping.")
                break

            # --- Altitude gating ---
            if altitude_min_m > 0.0 and current_altitude_m < altitude_min_m:
                if camera_frame_idx <= 3:
                    print(f"  [DBG] Camera frame #{camera_frame_idx} at {elapsed_s:.1f}s — "
                          f"altitude gated (alt={current_altitude_m:.1f}m < {altitude_min_m}m)")
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
            viz.clear_fine()

            # Build coarse obs with ENU centers
            coarse_obs = []
            coarse_top_k_enu = []
            for name, sim in zip(coarse.top_k_names, coarse.top_k_sims):
                e, n = obs.get_patch_center_enu(name)
                coarse_obs.append((e, n, sim))
                coarse_top_k_enu.append((e, n))

            pf.update_coarse(coarse_obs, altitude_m=current_altitude_m)

            # Continuous coarse trust teleport (skip when RTK-initialized)
            if coarse_obs and not rtk_init:
                top_e, top_n, top_sim = coarse_obs[0]
                coarse_decision = evaluate_coarse_trust(
                    top_sim, pf.weighted_spread(),
                    trust_tracker.recon_sim_ema, trust_config,
                )
                if coarse_decision is not None:
                    tp_frac, tp_sigma = coarse_decision
                    pf.inject_coarse_trust(
                        top_e, top_n, top_sim,
                        teleport_fraction_override=tp_frac,
                        teleport_sigma_override=tp_sigma,
                    )

            # --- Reconstruct satellite mosaic (single call for fine matching + viz) ---
            fp_recon = None
            if est_lat_prev is not None:
                # Use the center-crop square size so satellite_crop matches the drone
                # image (which is center-cropped from cam_w×cam_h → crop×crop → 320×320)
                cam_crop = min(cam_w, cam_h)
                fp_recon = reconstructor.reconstruct(
                    est_lat_prev, est_lon_prev, current_altitude_m,
                    est_hdg_prev + heading_offset_deg,
                    cam_fx, cam_fy, cam_crop, cam_crop,
                    output_size=(320, 320),
                )

            # --- Trust: coarse sims + reconstructed similarity ---
            top1_sim = coarse.top_k_sims[0] if coarse.top_k_sims else 0.0
            top2_sim = coarse.top_k_sims[1] if len(coarse.top_k_sims) > 1 else 0.0
            coarse_gap = top1_sim - top2_sim
            recon_sim = 0.0
            if fp_recon is not None:
                recon_sim = obs.compute_similarity(frame_bgr, fp_recon.satellite_crop)
                viz.satellite_footprint = fp_recon.satellite_crop
                viz.footprint_confidence = recon_sim
                viz.footprint_info_str = (
                    f"{fp_recon.footprint_w_m:.0f}x{fp_recon.footprint_h_m:.0f}m "
                    f"csim={top1_sim:.2f} rsim={recon_sim:.2f}"
                )
            else:
                viz.satellite_footprint = None

            # --- Fine match (adaptive) ---
            frame_trust = None
            fine_result = None
            sat_fine = None
            mosaic_fine = None
            patch_fine = None
            if pf.should_run_fine():
                fine_attempted += 1
                obs.altitude_m = current_altitude_m  # keep PnP altitude init accurate

                # A) Fine match on satellite crop (perspective-warped to drone FOV)
                if (fp_recon is not None and fp_recon.warp_M_inv is not None):
                    sat_fine = obs.fine_match_on_satellite(
                        frame_bgr,
                        fp_recon.satellite_crop,
                        fp_recon.mosaic_meta,
                        fp_recon.warp_M_inv,
                    )

                # B) Fine match on heading-rotated mosaic
                if (fp_recon is not None and
                        fp_recon.mosaic_rotated is not None and
                        fp_recon.rotation_center_px is not None):
                    mosaic_fine = obs.fine_match_on_mosaic(
                        frame_bgr,
                        fp_recon.mosaic_rotated,
                        fp_recon.mosaic_meta,
                        fp_recon.rotation_center_px,
                        fp_recon.heading_deg,
                    )

                # C) Fine match on coarse top-1 patch
                if coarse.top_k_names:
                    ctx_frac = pf.get_context_fraction()
                    patch_fine = obs.fine_match(frame_bgr, coarse.top_k_names[0],
                                               context_fraction=ctx_frac)

                # Store results for visualization
                best_mosaic = None
                if sat_fine is not None and mosaic_fine is not None:
                    best_mosaic = sat_fine if sat_fine.inliers >= mosaic_fine.inliers else mosaic_fine
                elif sat_fine is not None:
                    best_mosaic = sat_fine
                elif mosaic_fine is not None:
                    best_mosaic = mosaic_fine

                if best_mosaic is not None and fp_recon is not None:
                    if best_mosaic.patch_name == "satellite":
                        viz.mosaic_ref_img = fp_recon.satellite_crop
                    else:
                        viz.mosaic_ref_img = fp_recon.mosaic_rotated
                    viz.mosaic_mkpts_drone = best_mosaic.mkpts_drone
                    viz.mosaic_mkpts_ref = best_mosaic.mkpts_patch
                    viz.mosaic_fine_method = f"{best_mosaic.method}({best_mosaic.patch_name})"
                    viz.mosaic_fine_inliers = best_mosaic.inliers

                if patch_fine is not None:
                    top1_ctx = obs._build_context_patch(coarse.top_k_names[0],
                                                        pf.get_context_fraction())
                    viz.top1_mkpts_drone = patch_fine.mkpts_drone
                    viz.top1_mkpts_ref = patch_fine.mkpts_patch
                    viz.top1_ref_img = top1_ctx[0] if top1_ctx is not None else viz.coarse_patch
                    viz.top1_fine_method = patch_fine.method
                    viz.top1_fine_inliers = patch_fine.inliers

                # --- Unified trust-gated fine update ---
                all_fine = []
                if sat_fine is not None:
                    fe_s, fn_s = enu.wgs84_to_enu(sat_fine.lat, sat_fine.lon)
                    if np.isfinite(fe_s) and np.isfinite(fn_s):
                        all_fine.append({
                            "east_m": fe_s, "north_m": fn_s,
                            "inliers": sat_fine.inliers,
                            "heading_deg": sat_fine.heading_deg,
                            "source": "satellite",
                            "flow_consistency": sat_fine.flow_consistency,
                            "flow_magnitude_cv": sat_fine.flow_magnitude_cv,
                            "inlier_ratio": sat_fine.inlier_ratio,
                        })
                if mosaic_fine is not None:
                    fe_m, fn_m = enu.wgs84_to_enu(mosaic_fine.lat, mosaic_fine.lon)
                    if np.isfinite(fe_m) and np.isfinite(fn_m):
                        all_fine.append({
                            "east_m": fe_m, "north_m": fn_m,
                            "inliers": mosaic_fine.inliers,
                            "heading_deg": mosaic_fine.heading_deg,
                            "source": "mosaic",
                            "flow_consistency": mosaic_fine.flow_consistency,
                            "flow_magnitude_cv": mosaic_fine.flow_magnitude_cv,
                            "inlier_ratio": mosaic_fine.inlier_ratio,
                        })
                if patch_fine is not None:
                    fe_p, fn_p = enu.wgs84_to_enu(patch_fine.lat, patch_fine.lon)
                    if np.isfinite(fe_p) and np.isfinite(fn_p):
                        all_fine.append({
                            "east_m": fe_p, "north_m": fn_p,
                            "inliers": patch_fine.inliers,
                            "heading_deg": patch_fine.heading_deg,
                            "source": coarse.top_k_names[0],
                            "flow_consistency": patch_fine.flow_consistency,
                            "flow_magnitude_cv": patch_fine.flow_magnitude_cv,
                            "inlier_ratio": patch_fine.inlier_ratio,
                        })

                frame_trust = trust_tracker.evaluate_frame(
                    fine_candidates=all_fine,
                    recon_sim=recon_sim,
                    top1_sim=top1_sim,
                    pf_east=est_e, pf_north=est_n,
                    pf_spread=pf.weighted_spread(),
                    lost_spread=pf_config.lost_spread_m,
                    altitude_m=current_altitude_m,
                    timestamp_s=elapsed_s,
                )

                if frame_trust.best is not None:
                    best_cs = frame_trust.best
                    fine_result = best_cs  # for logging below
                    eff_sigma = trust_tracker.get_effective_sigma(
                        best_cs.confidence, pf_config.sigma_obs_fine,
                        is_static=pf.is_static)
                    eff_kappa = trust_tracker.get_effective_kappa(
                        best_cs.confidence, is_static=pf.is_static)
                    viz.fine_matched_name = best_cs.source
                    viz.footprint_confidence = best_cs.confidence

                    accepted = pf.update_fine(
                        best_cs.east_m, best_cs.north_m, best_cs.inliers,
                        best_cs.heading_deg,
                        sigma_override=eff_sigma,
                        kappa_override=eff_kappa,
                    )
                    if accepted:
                        fine_succeeded += 1
                    else:
                        viz.fine_matched_name = "REJECTED"
                else:
                    viz.fine_matched_name = "NO_TRUST"

            # --- Global correction: relocalize from coarse+fine when recon diverges ---
            global_corr = None
            coarse_fine_e, coarse_fine_n = None, None
            coarse_fine_inliers, coarse_fine_hdg = 0, None
            if patch_fine is not None:
                _cfe, _cfn = enu.wgs84_to_enu(patch_fine.lat, patch_fine.lon)
                if np.isfinite(_cfe) and np.isfinite(_cfn):
                    coarse_fine_e, coarse_fine_n = _cfe, _cfn
                    coarse_fine_inliers = patch_fine.inliers
                    coarse_fine_hdg = patch_fine.heading_deg

            global_corr = trust_tracker.evaluate_global_correction(
                recon_sim=recon_sim,
                coarse_fine_east=coarse_fine_e,
                coarse_fine_north=coarse_fine_n,
                coarse_fine_inliers=coarse_fine_inliers,
                coarse_fine_heading=coarse_fine_hdg,
                pf_east=est_e, pf_north=est_n,
                pf_spread=pf.weighted_spread(),
            )
            if global_corr is not None:
                pf.apply_global_correction(
                    global_corr.east_m, global_corr.north_m,
                    global_corr.heading_deg,
                    teleport_fraction=trust_config.global_corr_teleport_fraction,
                    teleport_sigma=trust_config.global_corr_teleport_sigma,
                )
                print(f"  [GLOBAL_CORRECTION] Teleported to ({global_corr.east_m:.1f}, "
                      f"{global_corr.north_m:.1f}) dist={global_corr.distance_from_pf_m:.1f}m "
                      f"inliers={global_corr.inliers_median} "
                      f"n_frames={global_corr.n_consistent_frames}")

            # Resample + transitions
            pf.resample_if_needed()
            pf.check_transitions()

            if pf.phase == Phase.TRACKING and tracking_start_ts is None:
                tracking_start_ts = ts_ns

            # Estimate
            est_e, est_n, est_hdg = pf.estimate()
            est_lat, est_lon = enu.enu_to_wgs84(est_e, est_n)
            est_lat_prev, est_lon_prev, est_hdg_prev = est_lat, est_lon, est_hdg

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

            # Extract trust info for logging
            _best_cs = frame_trust.best if frame_trust else None
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
                "fine_source": _best_cs.source if _best_cs else "",
                "fine_inliers": _best_cs.inliers if _best_cs else 0,
                "fine_confidence": _best_cs.confidence if _best_cs else 0.0,
                "fine_sigma_scale": trust_tracker.get_sigma_scale(_best_cs.confidence) if _best_cs else 0.0,
                "inlier_score": _best_cs.inlier_score if _best_cs else 0.0,
                "sim_score": _best_cs.sim_score if _best_cs else 0.0,
                "consistency_score": _best_cs.consistency_score if _best_cs else 0.0,
                "agreement_score": _best_cs.agreement_score if _best_cs else 0.0,
                "altitude_score": _best_cs.altitude_score if _best_cs else 0.0,
                "temporal_score": _best_cs.temporal_score if _best_cs else 0.0,
                "geometry_score": _best_cs.geometry_score if _best_cs else 0.0,
                "recon_sim": recon_sim,
                "top1_sim": top1_sim,
                "coarse_gap": coarse_gap,
                "pf_self_confidence": trust_tracker.pf_self_confidence_ema if frame_trust else 0.0,
                "drift_detected": frame_trust.drift_detected if frame_trust else False,
                "n_fine_candidates": len(frame_trust.all_scores) if frame_trust else 0,
                "recon_sim_ema": frame_trust.recon_sim_ema if frame_trust else 0.0,
                "global_correction": global_corr is not None,
                "recon_diverge_count": trust_tracker._recon_diverge_count,
            })

            # Build cache frame (v2: includes all fine candidates + trust signals)
            if save_cache:
                # Best fine result for backward compat
                fine_cache = None
                if _best_cs is not None:
                    fine_cache = {
                        "east_m": _best_cs.east_m,
                        "north_m": _best_cs.north_m,
                        "inliers": _best_cs.inliers,
                        "heading_deg": _best_cs.heading_deg,
                        "method": _best_cs.source,
                        "patch_name": _best_cs.source,
                    }
                # All fine candidates
                all_fine_cache = []
                if sat_fine is not None:
                    _fe, _fn = enu.wgs84_to_enu(sat_fine.lat, sat_fine.lon)
                    all_fine_cache.append({
                        "east_m": _fe, "north_m": _fn,
                        "inliers": sat_fine.inliers,
                        "heading_deg": sat_fine.heading_deg,
                        "method": sat_fine.method,
                        "patch_name": sat_fine.patch_name,
                        "source": "satellite",
                        "flow_consistency": sat_fine.flow_consistency,
                        "flow_magnitude_cv": sat_fine.flow_magnitude_cv,
                        "inlier_ratio": sat_fine.inlier_ratio,
                        "flow_heading_deg": sat_fine.flow_heading_deg,
                        "n_total_matches": sat_fine.n_total_matches,
                    })
                if mosaic_fine is not None:
                    _fe, _fn = enu.wgs84_to_enu(mosaic_fine.lat, mosaic_fine.lon)
                    all_fine_cache.append({
                        "east_m": _fe, "north_m": _fn,
                        "inliers": mosaic_fine.inliers,
                        "heading_deg": mosaic_fine.heading_deg,
                        "method": mosaic_fine.method,
                        "patch_name": mosaic_fine.patch_name,
                        "source": "mosaic",
                        "flow_consistency": mosaic_fine.flow_consistency,
                        "flow_magnitude_cv": mosaic_fine.flow_magnitude_cv,
                        "inlier_ratio": mosaic_fine.inlier_ratio,
                        "flow_heading_deg": mosaic_fine.flow_heading_deg,
                        "n_total_matches": mosaic_fine.n_total_matches,
                    })
                if patch_fine is not None:
                    _fe, _fn = enu.wgs84_to_enu(patch_fine.lat, patch_fine.lon)
                    all_fine_cache.append({
                        "east_m": _fe, "north_m": _fn,
                        "inliers": patch_fine.inliers,
                        "heading_deg": patch_fine.heading_deg,
                        "method": patch_fine.method,
                        "patch_name": patch_fine.patch_name,
                        "source": coarse.top_k_names[0],
                        "flow_consistency": patch_fine.flow_consistency,
                        "flow_magnitude_cv": patch_fine.flow_magnitude_cv,
                        "inlier_ratio": patch_fine.inlier_ratio,
                        "flow_heading_deg": patch_fine.flow_heading_deg,
                        "n_total_matches": patch_fine.n_total_matches,
                    })
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
                    "all_fine_results": all_fine_cache,
                    "recon_sim": recon_sim,
                    "top1_sim": top1_sim,
                    "coarse_gap": coarse_gap,
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
            "__cache_version__": 3,
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
    trust_config: TrustConfig,
    lock_cfg: dict,
    altitude_min_m: float,
    output_csv: str,
    output_plot: str,
    rtk_noise_m: float = 0.0,
) -> dict:
    """PF-only replay from a match cache file. No TRT, no MCAP."""
    print(f"[cache] Loading: {cache_path}")
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    header = payload[0]
    cache_frames = payload[1:]
    bag_name = header.get("bag_name", Path(cache_path).parent.name)
    enu = ENUFrame(header["enu_origin"]["lat"], header["enu_origin"]["lon"])
    cache_version = header.get("__cache_version__", 1)

    print(f"[{bag_name}|cache] {len(cache_frames)} frames  v={cache_version}  "
          f"(created {header.get('created_at', '?')})")

    lock_n_frames   = lock_cfg.get("lock_n_frames", 4)
    lock_sim_thresh = lock_cfg.get("lock_sim_threshold", 0.40)

    pf = ParticleFilter(pf_config, rng_seed=42)
    trust_tracker = TrustTracker(trust_config)
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

        # Apply RTK deltas (even for gated frames)
        for d in frame.get("rtk_deltas", []):
            if pf.particles is not None:
                delta = MotionDelta(**d)
                if rtk_noise_m > 0.0:
                    delta = MotionDelta(
                        dx_m=delta.dx_m + np.random.normal(0, rtk_noise_m),
                        dy_m=delta.dy_m + np.random.normal(0, rtk_noise_m),
                        heading_deg=delta.heading_deg + np.random.normal(0, rtk_noise_m * 2.0),
                        dt_s=delta.dt_s,
                    )
                pf.predict(delta)

        if frame.get("gated_out"):
            continue

        if altitude_min_m > 0.0 and alt < altitude_min_m:
            continue

        if not initialized:
            if pf.try_init(alt):
                initialized = True
                print(f"  [{bag_name}|cache] Altitude init: {alt:.1f}m")
            else:
                continue

        # RTK-init: seed PF from GT position (mirrors live RTK-init path)
        if pf.phase == Phase.UNINIT and gt_lat != 0.0 and gt_lon != 0.0:
            rtk_e, rtk_n = enu.wgs84_to_enu(gt_lat, gt_lon)
            pf.seed_from_position(rtk_e, rtk_n, heading_deg=0.0,
                                  sigma_pos=5.0, sigma_hdg=10.0)
            print(f"  [{bag_name}|cache] RTK-init: seeded at ({gt_lat:.6f}, {gt_lon:.6f})")

        coarse_names = frame.get("coarse_top_k_names", [])
        coarse_sims  = frame.get("coarse_top_k_sims",  [])
        coarse_enu   = frame.get("coarse_top_k_enu",   [])

        # Pre-seed lock logic (fallback if RTK-init didn't fire)
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
                pf.seed_from_coarse(coarse_enu, coarse_sims, sigma_override=tight_sigma)
                print(f"  [{bag_name}|cache] PF seeded: sigma={tight_sigma:.1f}m")
            continue

        # Coarse update
        coarse_obs = [(e, n, s) for (e, n), s in zip(coarse_enu, coarse_sims)]
        if coarse_obs:
            pf.update_coarse(coarse_obs, altitude_m=alt)

        # Coarse trust teleport
        if coarse_obs:
            top1_sim_val = coarse_sims[0] if coarse_sims else 0.0
            coarse_decision = evaluate_coarse_trust(
                top1_sim_val, pf.weighted_spread(),
                trust_tracker.recon_sim_ema, trust_config,
            )
            if coarse_decision is not None:
                tp_frac, tp_sigma = coarse_decision
                top_e, top_n_coord = coarse_enu[0]
                pf.inject_coarse_trust(
                    top_e, top_n_coord, top1_sim_val,
                    teleport_fraction_override=tp_frac,
                    teleport_sigma_override=tp_sigma,
                )

        # Fine update with trust model (skip while static before first motion)
        run_fine = pf.should_run_fine() and pf._motion_detected
        if run_fine:
            # Build candidates from cache
            all_fine_cached = frame.get("all_fine_results", [])
            recon_sim = frame.get("recon_sim", 0.0)
            top1_sim_val = frame.get("top1_sim", coarse_sims[0] if coarse_sims else 0.0)

            fine_candidates = []
            for fr in all_fine_cached:
                fine_candidates.append({
                    "east_m": fr["east_m"],
                    "north_m": fr["north_m"],
                    "inliers": fr["inliers"],
                    "heading_deg": fr.get("heading_deg"),
                    "source": fr.get("source", fr.get("patch_name", "")),
                    "flow_consistency": fr.get("flow_consistency", 0.0),
                    "flow_magnitude_cv": fr.get("flow_magnitude_cv", 1.0),
                    "inlier_ratio": fr.get("inlier_ratio", 0.0),
                })

            # Fallback for v1 caches
            if not fine_candidates:
                fc = frame.get("fine_result")
                if fc:
                    fine_candidates.append({
                        "east_m": fc["east_m"],
                        "north_m": fc["north_m"],
                        "inliers": fc["inliers"],
                        "heading_deg": fc.get("heading_deg"),
                        "source": fc.get("method", ""),
                    })

            if fine_candidates:
                fine_attempted += 1
                est_e, est_n, _ = pf.estimate()
                ft = trust_tracker.evaluate_frame(
                    fine_candidates=fine_candidates,
                    recon_sim=recon_sim,
                    top1_sim=top1_sim_val,
                    pf_east=est_e, pf_north=est_n,
                    pf_spread=pf.weighted_spread(),
                    lost_spread=pf_config.lost_spread_m,
                    altitude_m=alt,
                    timestamp_s=elapsed,
                )
                if ft.best is not None:
                    eff_sigma = trust_tracker.get_effective_sigma(
                        ft.best.confidence, pf_config.sigma_obs_fine,
                        is_static=pf.is_static)
                    eff_kappa = trust_tracker.get_effective_kappa(
                        ft.best.confidence, is_static=pf.is_static)
                    pf.update_fine(
                        ft.best.east_m, ft.best.north_m, ft.best.inliers,
                        ft.best.heading_deg,
                        sigma_override=eff_sigma,
                        kappa_override=eff_kappa,
                    )
                    fine_succeeded += 1
            else:
                fine_attempted += 1

        # --- Global correction from cached coarse-patch fine ---
        global_corr = None
        recon_sim_cache = frame.get("recon_sim", 0.0) if run_fine else 0.0
        coarse_fine_e_c, coarse_fine_n_c = None, None
        coarse_fine_inliers_c, coarse_fine_hdg_c = 0, None
        if run_fine:
            all_fine_cached_gc = frame.get("all_fine_results", [])
            for fr in all_fine_cached_gc:
                src = fr.get("source", fr.get("patch_name", ""))
                if src not in ("satellite", "mosaic") and fr["inliers"] > 0:
                    coarse_fine_e_c = fr["east_m"]
                    coarse_fine_n_c = fr["north_m"]
                    coarse_fine_inliers_c = fr["inliers"]
                    coarse_fine_hdg_c = fr.get("heading_deg")
                    break

        est_e_gc, est_n_gc, _ = pf.estimate()
        global_corr = trust_tracker.evaluate_global_correction(
            recon_sim=recon_sim_cache,
            coarse_fine_east=coarse_fine_e_c,
            coarse_fine_north=coarse_fine_n_c,
            coarse_fine_inliers=coarse_fine_inliers_c,
            coarse_fine_heading=coarse_fine_hdg_c,
            pf_east=est_e_gc, pf_north=est_n_gc,
            pf_spread=pf.weighted_spread(),
        )
        if global_corr is not None:
            pf.apply_global_correction(
                global_corr.east_m, global_corr.north_m,
                global_corr.heading_deg,
                teleport_fraction=trust_config.global_corr_teleport_fraction,
                teleport_sigma=trust_config.global_corr_teleport_sigma,
            )
            print(f"  [{bag_name}|cache|GLOBAL_CORRECTION] dist={global_corr.distance_from_pf_m:.1f}m "
                  f"inliers={global_corr.inliers_median} n_frames={global_corr.n_consistent_frames}")

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
            "fine_source": "",
            "fine_inliers": 0,
            "global_correction": global_corr is not None,
            "recon_diverge_count": trust_tracker._recon_diverge_count,
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
    parser.add_argument("--no-rtk-init", dest="rtk_init", action="store_false",
                        help="Disable RTK init (use coarse-lock instead)")
    parser.set_defaults(rtk_init=True)
    parser.add_argument("--rtk-noise", type=float, default=0.0,
                        help="Add Gaussian noise to RTK deltas (sigma in meters, 0=off)")
    args = parser.parse_args()
    run_replay(
        args.config,
        show_window=args.show,
        save_frames=args.save_frames,
        save_cache=args.save_cache,
        cache_path=args.use_cache,
        altitude_min_override=args.altitude_min,
        max_frames=args.max_frames,
        rtk_init=args.rtk_init,
        rtk_noise_m=args.rtk_noise,
    )
