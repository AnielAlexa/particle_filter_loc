#!/usr/bin/env python3
"""Grid sweep of PF params with fine_every_n_frames=3, using cache replay.
Parallelized across combos with multiprocessing."""

import copy
import csv
import io
import itertools
import os
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_csv_errors(csv_path):
    errors = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("state") == "TRACKING":
                e = float(row["error_m"])
                if e >= 0:
                    errors.append(e)
    return errors


def _run_one_combo(args):
    """Worker: run one param combo across all bags. Returns (params, bag_results)."""
    combo_idx, params, baseline_pf, trust_cfg_dict, bags, lock_cfg, alt_min = args

    # Lazy imports inside worker to avoid pickling issues
    from scripts.replay_mcap import _replay_from_cache
    from particle_filter_loc.particle_filter import PFConfig
    from particle_filter_loc.trust_model import TrustConfig

    # Suppress stdout in workers
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        pf_dict = copy.deepcopy(baseline_pf)
        for k, v in params.items():
            if k == "sigma_min_scale":
                continue
            pf_dict[k] = v
        pf_cfg = PFConfig(**pf_dict)

        t_dict = copy.deepcopy(trust_cfg_dict)
        t_dict["sigma_min_scale"] = params["sigma_min_scale"]
        t_cfg = TrustConfig(**t_dict)

        bag_errors = []
        for bag_name, cache_path in bags:
            # Use unique tmp file per worker+bag
            out_csv = os.path.join(tempfile.gettempdir(), f"tune_{combo_idx}_{bag_name}.csv")
            try:
                _replay_from_cache(
                    cache_path=cache_path,
                    pf_config=pf_cfg,
                    trust_config=t_cfg,
                    lock_cfg=lock_cfg,
                    altitude_min_m=alt_min,
                    output_csv=out_csv,
                    output_plot="",
                )
                errs = read_csv_errors(out_csv)
                if errs:
                    bag_errors.append((bag_name, float(np.median(errs)), float(np.percentile(errs, 90))))
                # Cleanup
                os.unlink(out_csv)
            except Exception:
                pass

        return (combo_idx, params, bag_errors)
    finally:
        sys.stdout = old_stdout


def main():
    bags_cfg = load_config(PKG_DIR / "config" / "bags.yaml")
    base_cfg = load_config(PKG_DIR / bags_cfg["base_config"])
    results_dir = PKG_DIR / "results"

    # Collect valid bags
    bags = []
    for bag in bags_cfg["bags"]:
        cache_path = results_dir / bag["name"] / "match_cache.pkl"
        if cache_path.exists():
            bags.append((bag["name"], str(cache_path)))
    print(f"Found {len(bags)} bags with caches")

    # Build trust config dict
    trust_cfg_dict = copy.deepcopy(base_cfg.get("trust", {}))
    trust_cfg_dict.pop("recon_sim_enabled", None)
    if "coarse_teleport_sigma_range" in trust_cfg_dict:
        v = trust_cfg_dict["coarse_teleport_sigma_range"]
        if isinstance(v, list):
            trust_cfg_dict["coarse_teleport_sigma_range"] = tuple(v)

    # Parameter grid
    param_grid = {
        "sigma_pos_tracking":    [0.5, 1.0, 1.5],
        "sigma_obs_fine":        [5.0, 10.0, 15.0],
        "roughen_scale":         [0.3, 0.5, 1.0],
        "likelihood_nu":         [3.0, 5.0, 10.0],
        "sigma_min_scale":       [0.2, 0.3, 0.5],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    baseline_pf = copy.deepcopy(base_cfg["particle_filter"])
    baseline_pf["fine_every_n_frames"] = 3

    lock_cfg = base_cfg.get("init", {})
    alt_min = base_cfg["replay"].get("altitude_min_process_m", 0)

    n_workers = min(cpu_count(), 4)  # Jetson Orin: leave headroom on 6-core CPU
    print(f"Grid: {len(combos)} combos × {len(bags)} bags = {len(combos) * len(bags)} runs")
    print(f"Using {n_workers} parallel workers\n")

    # Build work items
    work = []
    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        work.append((ci, params, baseline_pf, trust_cfg_dict, bags, lock_cfg, alt_min))

    # Run in parallel
    all_results = []
    done = 0
    with Pool(n_workers) as pool:
        for combo_idx, params, bag_errors in pool.imap_unordered(_run_one_combo, work):
            done += 1
            if bag_errors:
                mean_med = np.mean([x[1] for x in bag_errors])
                mean_p90 = np.mean([x[2] for x in bag_errors])
                worst_med = max(x[1] for x in bag_errors)
                all_results.append((params, mean_med, mean_p90, worst_med, bag_errors))

            if done % 20 == 0 or done == len(combos):
                print(f"  [{done}/{len(combos)}] completed...")

    # Sort by mean median
    all_results.sort(key=lambda x: x[1])

    print(f"\n{'='*80}")
    print(f"  TOP 20 CONFIGURATIONS (by mean median error)")
    print(f"{'='*80}")
    print(f"{'rank':>4s}  {'sig_pos':>7s} {'sig_fine':>8s} {'rough':>6s} {'nu':>5s} {'sig_min':>7s}  "
          f"{'mean_med':>8s} {'mean_p90':>8s} {'worst':>6s}")
    print(f"{'-'*4}  {'-'*7} {'-'*8} {'-'*6} {'-'*5} {'-'*7}  {'-'*8} {'-'*8} {'-'*6}")

    for i, (params, mm, mp, wm, _) in enumerate(all_results[:20]):
        print(f"{i+1:4d}  {params['sigma_pos_tracking']:7.1f} {params['sigma_obs_fine']:8.1f} "
              f"{params['roughen_scale']:6.1f} {params['likelihood_nu']:5.1f} {params['sigma_min_scale']:7.1f}  "
              f"{mm:7.1f}m {mp:7.1f}m {wm:5.1f}m")

    # Show worst 5 for contrast
    print(f"\n  WORST 5:")
    for i, (params, mm, mp, wm, _) in enumerate(all_results[-5:]):
        print(f"  {params['sigma_pos_tracking']:7.1f} {params['sigma_obs_fine']:8.1f} "
              f"{params['roughen_scale']:6.1f} {params['likelihood_nu']:5.1f} {params['sigma_min_scale']:7.1f}  "
              f"{mm:7.1f}m {mp:7.1f}m {wm:5.1f}m")

    # Show current baseline for comparison
    print(f"\n  Current config (fine_every=3): sig_pos=1.0 sig_fine=10.0 rough=0.5 nu=5.0 sig_min=0.3")

    # Show best per-bag breakdown
    if all_results:
        best = all_results[0]
        print(f"\n  Best config per-bag breakdown:")
        for name, med, p90 in best[3]:
            print(f"    {name:<12s}  median={med:.1f}m  p90={p90:.1f}m")


if __name__ == "__main__":
    main()
