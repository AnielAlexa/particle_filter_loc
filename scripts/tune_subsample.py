#!/usr/bin/env python3
"""Sweep camera_subsample and PF params using cache replay.

Cache replay doesn't subsample camera frames directly (all frames are in cache).
Instead we simulate subsampling by skipping frames in the cache replay loop.
We modify _replay_from_cache to accept a subsample parameter.
"""

import copy
import io
import sys
import contextlib
from pathlib import Path

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(PKG_DIR / "scripts"))

from replay_mcap import load_config


@contextlib.contextmanager
def suppress():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def run_cached_subsample(base_cfg, bag, cache_dir, overrides, subsample, rtk_noise=0.0):
    """Run cache replay with camera subsampling simulated by skipping frames."""
    import pickle
    import csv
    from particle_filter_loc.geo_utils import ENUFrame, haversine_m
    from particle_filter_loc.motion_model import MotionDelta
    from particle_filter_loc.particle_filter import PFConfig, ParticleFilter, Phase
    from particle_filter_loc.trust_model import TrustConfig, TrustTracker, evaluate_coarse_trust

    cfg = copy.deepcopy(base_cfg)
    for section, kvs in overrides.items():
        if section in cfg:
            for k, v in kvs.items():
                cfg[section][k] = v

    pf_cfg_dict = cfg["particle_filter"]
    trust_cfg_dict = cfg.get("trust", {})

    _trust_keys = {"trust_recon_sim_min", "trust_coarse_sim_min", "trust_high_inliers",
                   "trust_cross_agree_m", "trust_drift_recon_sim", "trust_drift_coarse_sim",
                   "coarse_trust_sim", "coarse_trust_fraction", "coarse_trust_sigma"}
    trust_config = TrustConfig(**trust_cfg_dict)
    pf_config = PFConfig(**{k: v for k, v in pf_cfg_dict.items() if k not in _trust_keys})

    cache_path = str(cache_dir / bag["name"] / "match_cache.pkl")
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    header = payload[0]
    cache_frames = payload[1:]
    enu = ENUFrame(header["enu_origin"]["lat"], header["enu_origin"]["lon"])
    lock_cfg = cfg.get("init", {})
    lock_n_frames = lock_cfg.get("lock_n_frames", 4)
    lock_sim_thresh = lock_cfg.get("lock_sim_threshold", 0.40)
    altitude_min_m = cfg["replay"].get("altitude_min_process_m", 50.0)

    pf = ParticleFilter(pf_config, rng_seed=42)
    trust_tracker = TrustTracker(trust_config)
    results = []
    initialized = False
    tracking_start_ts = None
    fine_attempted = fine_succeeded = 0
    _lock_patch = ""
    _lock_count = 0
    _lock_sims = []
    t_start_ns = cache_frames[0]["timestamp_ns"] if cache_frames else 0
    cam_frame_idx = 0

    for frame in cache_frames:
        ts_ns = frame["timestamp_ns"]
        alt = frame["altitude_m"]
        gt_lat = frame["gt_lat"]
        gt_lon = frame["gt_lon"]
        elapsed = frame.get("elapsed_s", (ts_ns - t_start_ns) * 1e-9)

        # Apply RTK deltas
        for d in frame.get("rtk_deltas", []):
            if pf.particles is not None:
                delta = MotionDelta(**d)
                if rtk_noise > 0.0:
                    delta = MotionDelta(
                        dx_m=delta.dx_m + np.random.normal(0, rtk_noise),
                        dy_m=delta.dy_m + np.random.normal(0, rtk_noise),
                        heading_deg=delta.heading_deg + np.random.normal(0, rtk_noise * 2.0),
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
            else:
                continue

        # RTK-init
        if pf.phase == Phase.UNINIT and gt_lat != 0.0 and gt_lon != 0.0:
            rtk_e, rtk_n = enu.wgs84_to_enu(gt_lat, gt_lon)
            pf.seed_from_position(rtk_e, rtk_n, heading_deg=0.0, sigma_pos=5.0, sigma_hdg=10.0)

        # Camera subsampling: skip frames
        cam_frame_idx += 1
        if cam_frame_idx % subsample != 0:
            # Still do resampling/transitions but skip coarse/fine matching
            pf.resample_if_needed()
            pf.check_transitions()
            if pf.phase != Phase.UNINIT:
                est_e, est_n, est_hdg = pf.estimate()
                est_lat, est_lon = enu.enu_to_wgs84(est_e, est_n)
                error_m = haversine_m(est_lat, est_lon, gt_lat, gt_lon) if gt_lat else -1.0
                results.append({
                    "error_m": error_m, "state": pf.phase.name,
                    "timestamp_ns": ts_ns,
                })
            continue

        coarse_names = frame.get("coarse_top_k_names", [])
        coarse_sims = frame.get("coarse_top_k_sims", [])
        coarse_enu = frame.get("coarse_top_k_enu", [])

        # Pre-seed lock
        if pf.phase == Phase.UNINIT:
            top1_name = coarse_names[0] if coarse_names else ""
            top1_sim = coarse_sims[0] if coarse_sims else 0.0
            if top1_name == _lock_patch and top1_sim >= lock_sim_thresh:
                _lock_count += 1
                _lock_sims.append(top1_sim)
            else:
                _lock_patch = top1_name
                _lock_count = 1
                _lock_sims = [top1_sim] if top1_sim >= lock_sim_thresh else []
            if _lock_count >= lock_n_frames and len(_lock_sims) >= lock_n_frames:
                tight_sigma = pf_config.sigma_obs_coarse / np.sqrt(_lock_count)
                pf.seed_from_coarse(coarse_enu, coarse_sims, sigma_override=tight_sigma)
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

        # Fine update
        run_fine = pf.should_run_fine() and pf._motion_detected
        if run_fine:
            all_fine_cached = frame.get("all_fine_results", [])
            recon_sim = frame.get("recon_sim", 0.0)
            top1_sim_val = frame.get("top1_sim", coarse_sims[0] if coarse_sims else 0.0)

            fine_candidates = []
            for fr in all_fine_cached:
                fine_candidates.append({
                    "east_m": fr["east_m"], "north_m": fr["north_m"],
                    "inliers": fr["inliers"],
                    "heading_deg": fr.get("heading_deg"),
                    "source": fr.get("source", fr.get("patch_name", "")),
                    "flow_consistency": fr.get("flow_consistency", 0.0),
                    "flow_magnitude_cv": fr.get("flow_magnitude_cv", 1.0),
                    "inlier_ratio": fr.get("inlier_ratio", 0.0),
                })

            if not fine_candidates:
                fc = frame.get("fine_result")
                if fc:
                    fine_candidates.append({
                        "east_m": fc["east_m"], "north_m": fc["north_m"],
                        "inliers": fc["inliers"],
                        "heading_deg": fc.get("heading_deg"),
                        "source": fc.get("method", ""),
                    })

            if fine_candidates:
                fine_attempted += 1
                est_e, est_n, _ = pf.estimate()
                ft = trust_tracker.evaluate_frame(
                    fine_candidates=fine_candidates,
                    recon_sim=recon_sim, top1_sim=top1_sim_val,
                    pf_east=est_e, pf_north=est_n,
                    pf_spread=pf.weighted_spread(),
                    lost_spread=pf_config.lost_spread_m,
                    altitude_m=alt, timestamp_s=elapsed,
                )
                if ft.best is not None:
                    eff_sigma = trust_tracker.get_effective_sigma(
                        ft.best.confidence, pf_config.sigma_obs_fine, is_static=pf.is_static)
                    eff_kappa = trust_tracker.get_effective_kappa(
                        ft.best.confidence, is_static=pf.is_static)
                    accepted = pf.update_fine(
                        ft.best.east_m, ft.best.north_m, ft.best.inliers,
                        ft.best.heading_deg, sigma_override=eff_sigma, kappa_override=eff_kappa,
                    )
                    if accepted:
                        fine_succeeded += 1

        # Global correction
        recon_sim_gc = frame.get("recon_sim", 0.0) if run_fine else 0.0
        coarse_fine_e, coarse_fine_n = None, None
        coarse_fine_inliers, coarse_fine_hdg = 0, None
        if run_fine:
            for fr in frame.get("all_fine_results", []):
                src = fr.get("source", fr.get("patch_name", ""))
                if src not in ("satellite", "mosaic") and fr["inliers"] > 0:
                    coarse_fine_e = fr["east_m"]
                    coarse_fine_n = fr["north_m"]
                    coarse_fine_inliers = fr["inliers"]
                    coarse_fine_hdg = fr.get("heading_deg")
                    break
        est_e_gc, est_n_gc, _ = pf.estimate()
        global_corr = trust_tracker.evaluate_global_correction(
            recon_sim=recon_sim_gc,
            coarse_fine_east=coarse_fine_e, coarse_fine_north=coarse_fine_n,
            coarse_fine_inliers=coarse_fine_inliers, coarse_fine_heading=coarse_fine_hdg,
            pf_east=est_e_gc, pf_north=est_n_gc, pf_spread=pf.weighted_spread(),
        )
        if global_corr is not None:
            pf.apply_global_correction(
                global_corr.east_m, global_corr.north_m, global_corr.heading_deg,
                teleport_fraction=trust_config.global_corr_teleport_fraction,
                teleport_sigma=trust_config.global_corr_teleport_sigma,
            )

        pf.resample_if_needed()
        pf.check_transitions()

        if pf.phase == Phase.TRACKING and tracking_start_ts is None:
            tracking_start_ts = ts_ns

        est_e, est_n, est_hdg = pf.estimate()
        est_lat, est_lon = enu.enu_to_wgs84(est_e, est_n)
        error_m = haversine_m(est_lat, est_lon, gt_lat, gt_lon) if gt_lat else -1.0

        results.append({
            "error_m": error_m, "state": pf.phase.name,
            "timestamp_ns": ts_ns,
        })

    if not results:
        return None

    errors = [r["error_m"] for r in results if r["error_m"] >= 0]
    tracking_errors = [r["error_m"] for r in results if r["error_m"] >= 0 and r["state"] == "TRACKING"]

    return {
        "n_frames": len(results),
        "median_err_all": float(np.median(errors)) if errors else -1,
        "median_err_tracking": float(np.median(tracking_errors)) if tracking_errors else -1,
        "p90_err": float(np.percentile(tracking_errors, 90)) if tracking_errors else (float(np.percentile(errors, 90)) if errors else -1),
        "fine_rate_pct": 100.0 * fine_succeeded / max(fine_attempted, 1),
    }


def run_sweep(base_cfg, bags, cache_dir, overrides, label, subsample=1, rtk_noise=0.0):
    meds, p90s, fines = [], [], []
    per_bag = {}
    for bag in bags:
        with suppress():
            s = run_cached_subsample(base_cfg, bag, cache_dir, overrides, subsample, rtk_noise)
        if s is None or s["n_frames"] == 0:
            continue
        med = s["median_err_tracking"] if s["median_err_tracking"] >= 0 else s["median_err_all"]
        p90 = s.get("p90_err", -1)
        fine = s.get("fine_rate_pct", 0)
        if med >= 0: meds.append(med)
        if p90 >= 0: p90s.append(p90)
        fines.append(fine)
        per_bag[bag["name"]] = {"med": med, "p90": p90, "fine": fine}

    return {
        "label": label, "n_bags": len(meds),
        "mean_med": np.mean(meds) if meds else -1,
        "mean_p90": np.mean(p90s) if p90s else -1,
        "mean_fine": np.mean(fines) if fines else 0,
        "per_bag": per_bag,
    }


def print_table(results, bag_names):
    bag_cols = "  ".join(f"{n:>7s}" for n in bag_names)
    hdr = f"{'config':<45s} {'med':>6s} {'p90':>6s} {'fine%':>5s}  {bag_cols}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        bv = []
        for n in bag_names:
            b = r["per_bag"].get(n)
            bv.append(f"{b['med']:7.1f}" if b and b["med"] >= 0 else f"{'—':>7s}")
        bs = "  ".join(bv)
        mm = f"{r['mean_med']:.1f}" if r["mean_med"] >= 0 else "—"
        mp = f"{r['mean_p90']:.1f}" if r["mean_p90"] >= 0 else "—"
        print(f"{r['label']:<45s} {mm:>6s} {mp:>6s} {r['mean_fine']:5.1f}  {bs}")


def main():
    bags_yaml = PKG_DIR / "config" / "bags.yaml"
    with open(bags_yaml) as f:
        bags_cfg = yaml.safe_load(f)
    base_cfg = load_config(str(PKG_DIR / bags_cfg["base_config"]))
    cache_dir = PKG_DIR / "results"
    bags = [b for b in bags_cfg["bags"]
            if (cache_dir / b["name"] / "match_cache.pkl").exists()]
    names = [b["name"] for b in bags]
    print(f"Bags: {names}")

    # combo_F is now the default in config, so {} = combo_F
    combo_f = {}

    # ══════════════════════════════════════════════════════════════════
    # SWEEP 1: subsample effect with combo_F, no noise
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("SWEEP 1: camera_subsample with combo_F (no RTK noise)")
    print(f"{'='*80}")

    results1 = []
    for sub in [1, 2, 3, 5]:
        label = f"sub={sub} noise=0"
        print(f"  Running {label}...", end="", flush=True)
        r = run_sweep(base_cfg, bags, cache_dir, combo_f, label, subsample=sub)
        mm = f"{r['mean_med']:.1f}m" if r["mean_med"] >= 0 else "—"
        print(f"  med={mm}")
        results1.append(r)
    print()
    print_table(results1, names)

    # ══════════════════════════════════════════════════════════════════
    # SWEEP 2: subsample=3 + noise levels
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("SWEEP 2: subsample=3 across noise levels (combo_F)")
    print(f"{'='*80}")

    results2 = []
    for noise in [0.0, 1.0, 2.0, 3.0, 5.0]:
        label = f"sub=3 noise={noise:.0f}m"
        print(f"  Running {label}...", end="", flush=True)
        r = run_sweep(base_cfg, bags, cache_dir, combo_f, label, subsample=3, rtk_noise=noise)
        mm = f"{r['mean_med']:.1f}m" if r["mean_med"] >= 0 else "—"
        print(f"  med={mm}")
        results2.append(r)
    print()
    print_table(results2, names)

    # ══════════════════════════════════════════════════════════════════
    # SWEEP 3: Tune PF params for subsample=3 at noise=2m
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("SWEEP 3: Tune params for subsample=3 at noise=2m")
    print(f"{'='*80}")

    results3 = []
    # Baseline combo_F at sub=3 noise=2
    r = run_sweep(base_cfg, bags, cache_dir, combo_f, "combo_F (baseline)", subsample=3, rtk_noise=2.0)
    print(f"  baseline: med={r['mean_med']:.1f}m")
    results3.append(r)

    # Higher drift noise
    for dn in [1.0, 1.5, 2.0, 3.0]:
        ov = {"particle_filter": {"drift_noise_m_per_s": dn, "drift_noise_hdg_per_s": dn * 1.5}}
        label = f"drift={dn}"
        r = run_sweep(base_cfg, bags, cache_dir, ov, label, subsample=3, rtk_noise=2.0)
        print(f"  {label}: med={r['mean_med']:.1f}m")
        results3.append(r)

    # More particles
    for nd, nt in [(300, 150), (500, 250), (800, 400)]:
        ov = {"particle_filter": {"n_dispersed": nd, "n_tracking": nt}}
        label = f"particles={nd}/{nt}"
        r = run_sweep(base_cfg, bags, cache_dir, ov, label, subsample=3, rtk_noise=2.0)
        print(f"  {label}: med={r['mean_med']:.1f}m")
        results3.append(r)

    # Process noise
    for sp in [0.5, 1.0, 2.0, 3.0]:
        ov = {"particle_filter": {"sigma_pos_tracking": sp}}
        label = f"proc_noise={sp}"
        r = run_sweep(base_cfg, bags, cache_dir, ov, label, subsample=3, rtk_noise=2.0)
        print(f"  {label}: med={r['mean_med']:.1f}m")
        results3.append(r)

    # Roughening
    for rs in [0.5, 1.0, 2.0, 3.0]:
        ov = {"particle_filter": {"roughen_scale": rs}}
        label = f"roughen={rs}"
        r = run_sweep(base_cfg, bags, cache_dir, ov, label, subsample=3, rtk_noise=2.0)
        print(f"  {label}: med={r['mean_med']:.1f}m")
        results3.append(r)

    # Fine sigma
    for sf in [3.0, 5.0, 8.0]:
        ov = {"particle_filter": {"sigma_obs_fine": sf}}
        label = f"sigma_fine={sf}"
        r = run_sweep(base_cfg, bags, cache_dir, ov, label, subsample=3, rtk_noise=2.0)
        print(f"  {label}: med={r['mean_med']:.1f}m")
        results3.append(r)

    # Fine gate
    for fg in [20.0, 40.0, 60.0, 0.0]:
        ov = {"particle_filter": {"fine_consistency_max_m": fg}}
        label = f"gate={fg:.0f}" + (" (off)" if fg == 0 else "")
        r = run_sweep(base_cfg, bags, cache_dir, ov, label, subsample=3, rtk_noise=2.0)
        print(f"  {label}: med={r['mean_med']:.1f}m")
        results3.append(r)

    print()
    print_table(results3, names)

    # ══════════════════════════════════════════════════════════════════
    # SWEEP 4: Best sub=3 config across noise levels
    # ══════════════════════════════════════════════════════════════════
    # Find best from sweep 3
    valid = [r for r in results3 if r["mean_med"] >= 0]
    best = min(valid, key=lambda r: r["mean_med"])
    print(f"\n>>> Best sub=3 config: {best['label']}  med={best['mean_med']:.1f}m")

    # Build best combined overrides
    # Try combining top individual winners
    print(f"\n{'='*80}")
    print("SWEEP 4: Combined configs for sub=3")
    print(f"{'='*80}")

    combos = [
        ("combo_F (default)", {}),
        ("drift2+proc1+roughen2", {"particle_filter": {
            "drift_noise_m_per_s": 2.0, "drift_noise_hdg_per_s": 3.0,
            "sigma_pos_tracking": 1.0, "roughen_scale": 2.0}}),
        ("drift2+proc2+roughen2", {"particle_filter": {
            "drift_noise_m_per_s": 2.0, "drift_noise_hdg_per_s": 3.0,
            "sigma_pos_tracking": 2.0, "roughen_scale": 2.0}}),
        ("drift3+proc2+roughen2+500p", {"particle_filter": {
            "drift_noise_m_per_s": 3.0, "drift_noise_hdg_per_s": 4.5,
            "sigma_pos_tracking": 2.0, "roughen_scale": 2.0,
            "n_dispersed": 500, "n_tracking": 250}}),
        ("drift2+proc1+roughen2+gate60", {"particle_filter": {
            "drift_noise_m_per_s": 2.0, "drift_noise_hdg_per_s": 3.0,
            "sigma_pos_tracking": 1.0, "roughen_scale": 2.0,
            "fine_consistency_max_m": 60.0}}),
    ]

    results4 = []
    for label, ov in combos:
        sub3_results = []
        for noise in [0.0, 2.0, 3.0, 5.0]:
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"{label}|n={noise:.0f}",
                          subsample=3, rtk_noise=noise)
            sub3_results.append(r)
        # Print compact summary
        vals = " | ".join(f"n={n:.0f}:{r['mean_med']:.1f}m" for n, r in
                          zip([0, 2, 3, 5], sub3_results))
        print(f"  {label}: {vals}")
        results4.extend(sub3_results)

    print()
    print_table(results4, names)


if __name__ == "__main__":
    main()
