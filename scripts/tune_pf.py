#!/usr/bin/env python3
"""Fast PF parameter grid search using match caches.

No TRT, no MCAP — loads cached coarse/fine match data and sweeps PF config
parameters in seconds.

Usage:
  python3 tune_pf.py \
      --caches results/Day2.6/match_cache.pkl results/Day3.1/match_cache.pkl \
      [--base-config config/pf_config.yaml] \
      [--output results/tune_results.csv] \
      [--metric median_err_tracking] \
      [--top-n 20]

After the run, top-N parameter combos are printed and saved. Apply the best
combo to pf_config.yaml and re-run bench_all_bags.py --use-cache to validate.
"""

import argparse
import csv
import itertools
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from particle_filter_loc.geo_utils import ENUFrame, haversine_m
from particle_filter_loc.motion_model import MotionDelta
from particle_filter_loc.particle_filter import PFConfig, ParticleFilter, Phase


# ---------------------------------------------------------------------------
# Grid definition — edit these to change the search space
# ---------------------------------------------------------------------------
GRID = {
    "sigma_obs_coarse":      [20.0, 30.0, 40.0, 50.0, 60.0],
    "sigma_obs_fine":        [5.0, 8.0, 12.0],
    "converge_spread_m":     [60.0, 80.0, 100.0],
    "tracking_spread_m":     [15.0, 20.0, 30.0],
    "fine_every_n_frames":   [2, 3, 5],
    "fine_consistency_max_m":[0.0, 50.0, 100.0, 150.0],
}
# Total: 5×3×3×3×3×4 = 1620 combos.
# Run on working bags only (Day3.1, Day4.2, Day4.3) — bags with no tracking
# (Day3.2, Day3.3) dominate the metric with 9999m sentinel and are excluded.


def _load_cache(path: str):
    """Load a match cache. Returns (header_dict, frames_list, ENUFrame)."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    header = payload[0]
    frames = payload[1:]
    enu = ENUFrame(header["enu_origin"]["lat"], header["enu_origin"]["lon"])
    print(f"  Loaded {Path(path).name}: {len(frames)} frames  "
          f"bag={header.get('bag_name','?')}  "
          f"created={header.get('created_at','?')[:10]}")
    return header, frames, enu


def pf_replay_from_cache(
    frames: list,
    enu: ENUFrame,
    pf_config: PFConfig,
    lock_n_frames: int = 4,
    lock_sim_thresh: float = 0.40,
    altitude_min_m: float = 50.0,
    rng_seed: int = 42,
) -> dict:
    """Pure-PF replay from cached match data. Returns summary dict.

    Uses rng_seed=42 by default for reproducibility across grid search.
    All motion predict steps are applied even for gated frames so the
    motion model stays coherent.
    """
    pf = ParticleFilter(pf_config, rng_seed=rng_seed)
    results = []
    initialized = False
    tracking_start_ns = None
    fine_attempted = fine_succeeded = 0
    _lock_patch = ""
    _lock_count = 0
    _lock_sims: list = []
    t_start_ns = frames[0]["timestamp_ns"] if frames else 0

    for frame in frames:
        ts_ns  = frame["timestamp_ns"]
        alt    = frame.get("altitude_m", 0.0)
        gt_lat = frame.get("gt_lat", 0.0)
        gt_lon = frame.get("gt_lon", 0.0)

        # Apply RTK deltas for motion continuity (even for gated/uninit frames)
        for d in frame.get("rtk_deltas", []):
            if pf.particles is not None:
                pf.predict(MotionDelta(**d))

        if frame.get("gated_out"):
            continue

        # Altitude gating
        if altitude_min_m > 0.0 and alt < altitude_min_m:
            continue

        # Init check
        if not initialized:
            if pf.try_init(alt):
                initialized = True
            else:
                continue

        coarse_names = frame.get("coarse_top_k_names", [])
        coarse_sims  = frame.get("coarse_top_k_sims",  [])
        coarse_enu   = frame.get("coarse_top_k_enu",   [])

        # Pre-seed coarse-lock
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
            continue

        # Coarse update
        coarse_obs = [(e, n, s) for (e, n), s in zip(coarse_enu, coarse_sims)]
        if coarse_obs:
            pf.update_coarse(coarse_obs, altitude_m=alt)

        # Fine update — use cache result if PF decides to run fine this frame
        fine_cache = frame.get("fine_result")
        run_fine = pf.should_run_fine()  # has side effect: increments frame count
        if fine_cache and run_fine:
            fe = fine_cache["east_m"]
            fn = fine_cache["north_m"]
            pf.update_fine(fe, fn, fine_cache["inliers"], fine_cache.get("heading_deg"))
            fine_attempted += 1
            fine_succeeded += 1
        elif run_fine:
            fine_attempted += 1

        pf.resample_if_needed()
        pf.check_transitions()

        if pf.phase == Phase.TRACKING and tracking_start_ns is None:
            tracking_start_ns = ts_ns

        est_e, est_n, _ = pf.estimate()
        est_lat, est_lon = enu.enu_to_wgs84(est_e, est_n)
        error_m = haversine_m(est_lat, est_lon, gt_lat, gt_lon) if gt_lat else -1.0

        results.append({
            "error_m": error_m,
            "state": pf.phase.name,
            "timestamp_ns": ts_ns,
        })

    # Compute metrics
    errors = [r["error_m"] for r in results if r["error_m"] >= 0]
    track_errs = [r["error_m"] for r in results
                  if r["error_m"] >= 0 and r["state"] == "TRACKING"]

    med_all   = float(np.median(errors))   if errors      else 9999.0
    med_track = float(np.median(track_errs)) if track_errs else 9999.0
    p90_track = float(np.percentile(track_errs, 90)) if track_errs else 9999.0

    converge_s = -1.0
    if tracking_start_ns:
        converge_s = (tracking_start_ns - t_start_ns) * 1e-9

    return {
        "n_frames": len(results),
        "n_tracking": len(track_errs),
        "median_err_all": med_all,
        "median_err_tracking": med_track,
        "p90_err_tracking": p90_track,
        "converge_s": converge_s,
        "fine_rate_pct": 100.0 * fine_succeeded / max(fine_attempted, 1),
    }


def _run_grid_search(
    caches: list,           # list of (header, frames, enu)
    base_pf_dict: dict,
    grid: dict,
    metric: str = "median_err_tracking",
    lock_n_frames: int = 4,
    lock_sim_thresh: float = 0.40,
    altitude_min_m: float = 50.0,
):
    """Run grid search. Returns list of result dicts sorted by metric."""
    param_names = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in param_names]))
    n_total = len(combos) * len(caches)
    print(f"\n[tune] Grid: {len(combos)} combos × {len(caches)} bags = {n_total} PF replays")
    print(f"[tune] Sorting by: {metric}")

    all_results = []
    t0 = time.monotonic()

    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        pf_dict = {**base_pf_dict, **params}

        # PFConfig only accepts its own fields; filter out unknowns
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(PFConfig)}
        pf_dict_clean = {k: v for k, v in pf_dict.items() if k in valid_fields}
        pf_config = PFConfig(**pf_dict_clean)

        bag_summaries = []
        for header, frames, enu in caches:
            s = pf_replay_from_cache(
                frames, enu, pf_config,
                lock_n_frames=lock_n_frames,
                lock_sim_thresh=lock_sim_thresh,
                altitude_min_m=altitude_min_m,
            )
            bag_summaries.append(s)

        # Aggregate across bags (weighted by frame count)
        total_frames = sum(s["n_frames"] for s in bag_summaries)
        weights = [s["n_frames"] / max(total_frames, 1) for s in bag_summaries]
        agg = {
            "median_err_tracking": sum(w * s["median_err_tracking"]
                                        for w, s in zip(weights, bag_summaries)),
            "p90_err_tracking": sum(w * s["p90_err_tracking"]
                                     for w, s in zip(weights, bag_summaries)),
            "median_err_all": sum(w * s["median_err_all"]
                                   for w, s in zip(weights, bag_summaries)),
            "converge_s": np.mean([s["converge_s"] for s in bag_summaries
                                   if s["converge_s"] >= 0]) if any(
                s["converge_s"] >= 0 for s in bag_summaries) else -1.0,
            "fine_rate_pct": np.mean([s["fine_rate_pct"] for s in bag_summaries]),
            "n_frames_total": total_frames,
        }
        all_results.append({**params, **agg})

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(combos):
            elapsed = time.monotonic() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            best_so_far = min(r[metric] for r in all_results)
            print(f"  [{i+1:4d}/{len(combos)}]  elapsed={elapsed:5.1f}s  "
                  f"ETA={eta:5.1f}s  best_{metric}={best_so_far:.2f}m")

    all_results.sort(key=lambda r: r[metric])
    return all_results


def _print_top_results(results: list, top_n: int, metric: str):
    param_names = list(GRID.keys())
    col_w = max(len(k) for k in param_names + ["metric_val"]) + 2

    print(f"\n{'='*80}")
    print(f"Top-{top_n} parameter combos by {metric}:")
    print(f"{'='*80}")
    header_parts = [f"{k:<{col_w}}" for k in param_names]
    header_parts += [f"{'med_trk':>9}", f"{'p90_trk':>9}", f"{'conv_s':>8}", f"{'fine%':>6}"]
    print("  " + " ".join(header_parts))
    print("  " + "-" * (sum(col_w for _ in param_names) + 40))

    for r in results[:top_n]:
        row_parts = [f"{r[k]:<{col_w}}" for k in param_names]
        row_parts += [
            f"{r['median_err_tracking']:>8.2f}m",
            f"{r['p90_err_tracking']:>8.2f}m",
            f"{r['converge_s']:>7.1f}s",
            f"{r['fine_rate_pct']:>5.1f}%",
        ]
        print("  " + " ".join(row_parts))

    print(f"\nBest combo:")
    best = results[0]
    for k in param_names:
        print(f"  {k}: {best[k]}")
    print(f"  → median_err_tracking = {best['median_err_tracking']:.2f}m  "
          f"p90 = {best['p90_err_tracking']:.2f}m")


def main():
    parser = argparse.ArgumentParser(description="PF parameter grid search from match caches")
    parser.add_argument("--caches", nargs="+", required=True,
                        help="Paths to match_cache.pkl files")
    parser.add_argument("--base-config", default=str(PKG_DIR / "config" / "pf_config.yaml"),
                        help="Base pf_config.yaml for non-grid PF parameters")
    parser.add_argument("--output", default=str(PKG_DIR / "results" / "tune_results.csv"),
                        help="Output CSV path")
    parser.add_argument("--metric", default="median_err_tracking",
                        choices=["median_err_tracking", "p90_err_tracking",
                                 "median_err_all", "converge_s"],
                        help="Metric to minimize")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top combos to display")
    parser.add_argument("--altitude-min", type=float, default=50.0,
                        help="Altitude threshold for frame gating (m)")
    args = parser.parse_args()

    # Load caches
    print("[tune] Loading caches...")
    caches = []
    for p in args.caches:
        if not Path(p).exists():
            print(f"  [WARN] Cache not found: {p} — skipping")
            continue
        caches.append(_load_cache(p))

    if not caches:
        print("[tune] No caches loaded. Run replay_mcap.py --save-cache first.")
        return

    # Load base PF config
    base_cfg = yaml.safe_load(open(args.base_config))
    base_pf_dict = base_cfg["particle_filter"]
    lock_cfg = base_cfg.get("init", {})

    # Baseline run (current params, no grid)
    print("\n[tune] Baseline run with current pf_config.yaml...")
    base_pf_config = PFConfig(**{k: v for k, v in base_pf_dict.items()
                                  if k in {f.name for f in __import__('dataclasses').fields(PFConfig)}})
    baseline_summaries = []
    for header, frames, enu in caches:
        s = pf_replay_from_cache(
            frames, enu, base_pf_config,
            lock_n_frames=lock_cfg.get("lock_n_frames", 4),
            lock_sim_thresh=lock_cfg.get("lock_sim_threshold", 0.40),
            altitude_min_m=args.altitude_min,
        )
        baseline_summaries.append((header.get("bag_name", "?"), s))
        print(f"  {header.get('bag_name','?'):12s}  "
              f"med_track={s['median_err_tracking']:.2f}m  "
              f"p90={s['p90_err_tracking']:.2f}m  "
              f"conv={s['converge_s']:.1f}s")

    # Grid search
    all_results = _run_grid_search(
        caches, base_pf_dict, GRID,
        metric=args.metric,
        lock_n_frames=lock_cfg.get("lock_n_frames", 4),
        lock_sim_thresh=lock_cfg.get("lock_sim_threshold", 0.40),
        altitude_min_m=args.altitude_min,
    )

    # Print top results
    _print_top_results(all_results, args.top_n, args.metric)

    # Save CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if all_results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[tune] Full results saved: {args.output}")

    print("\n[tune] To apply best params, update pf_config.yaml particle_filter section,")
    print("       then run: python3 bench_all_bags.py --bags config/bags.yaml --use-cache")


if __name__ == "__main__":
    main()
