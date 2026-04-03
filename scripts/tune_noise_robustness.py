#!/usr/bin/env python3
"""Test noise robustness: sweep rtk_noise + drift_bias for best vs current config."""

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


def _run_one(args):
    """Worker: run one (config, noise, bag) combo."""
    cfg_name, pf_dict, trust_dict, bag_name, cache_path, lock_cfg, alt_min, noise_m, drift_m_s, seed = args

    from scripts.replay_mcap import _replay_from_cache
    from particle_filter_loc.particle_filter import PFConfig
    from particle_filter_loc.trust_model import TrustConfig

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        np.random.seed(seed)
        pf_cfg = PFConfig(**pf_dict)
        t_cfg = TrustConfig(**trust_dict)
        out_csv = os.path.join(tempfile.gettempdir(),
                               f"noise_{cfg_name}_{bag_name}_n{noise_m}_d{drift_m_s}_s{seed}.csv")
        _replay_from_cache(
            cache_path=cache_path,
            pf_config=pf_cfg,
            trust_config=t_cfg,
            lock_cfg=lock_cfg,
            altitude_min_m=alt_min,
            output_csv=out_csv,
            output_plot="",
            rtk_noise_m=noise_m,
            drift_bias_m_per_s=drift_m_s,
        )
        errs = read_csv_errors(out_csv)
        os.unlink(out_csv)
        if errs:
            return (cfg_name, bag_name, noise_m, drift_m_s, seed,
                    float(np.median(errs)), float(np.percentile(errs, 90)))
        return None
    except Exception:
        return None
    finally:
        sys.stdout = old_stdout


def main():
    bags_cfg = load_config(PKG_DIR / "config" / "bags.yaml")
    base_cfg = load_config(PKG_DIR / bags_cfg["base_config"])
    results_dir = PKG_DIR / "results"

    bags = []
    for bag in bags_cfg["bags"]:
        cache_path = results_dir / bag["name"] / "match_cache.pkl"
        if cache_path.exists():
            bags.append((bag["name"], str(cache_path)))

    trust_cfg_dict = copy.deepcopy(base_cfg.get("trust", {}))
    trust_cfg_dict.pop("recon_sim_enabled", None)
    if "coarse_teleport_sigma_range" in trust_cfg_dict:
        v = trust_cfg_dict["coarse_teleport_sigma_range"]
        if isinstance(v, list):
            trust_cfg_dict["coarse_teleport_sigma_range"] = tuple(v)

    lock_cfg = base_cfg.get("init", {})
    alt_min = base_cfg["replay"].get("altitude_min_process_m", 0)

    # Two configs to compare
    configs = {}

    # Current: fine_every=3, sigma_pos=1.0, sigma_obs_fine=10.0
    current_pf = copy.deepcopy(base_cfg["particle_filter"])
    current_pf["fine_every_n_frames"] = 3
    configs["current"] = (current_pf, copy.deepcopy(trust_cfg_dict))

    # Best: fine_every=3, sigma_pos=0.5, sigma_obs_fine=15.0
    best_pf = copy.deepcopy(base_cfg["particle_filter"])
    best_pf["fine_every_n_frames"] = 3
    best_pf["sigma_pos_tracking"] = 0.5
    best_pf["sigma_obs_fine"] = 15.0
    configs["tuned"] = (best_pf, copy.deepcopy(trust_cfg_dict))

    # Noise levels
    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0]
    drift_levels = [0.0, 0.1, 0.3, 0.5, 1.0]  # m/s constant drift
    n_seeds = 3  # average over random seeds for noise>0

    # Build work items
    work = []
    for cfg_name, (pf_dict, t_dict) in configs.items():
        for noise_m in noise_levels:
            for drift_m_s in drift_levels:
                if noise_m == 0 and drift_m_s == 0:
                    seeds = [42]  # deterministic
                else:
                    seeds = list(range(n_seeds))
                for seed in seeds:
                    for bag_name, cache_path in bags:
                        work.append((cfg_name, pf_dict, t_dict, bag_name, cache_path,
                                     lock_cfg, alt_min, noise_m, drift_m_s, seed))

    n_workers = min(cpu_count(), 4)  # Jetson Orin: leave headroom on 6-core CPU
    print(f"Configs: {list(configs.keys())}")
    print(f"Noise levels: {noise_levels}")
    print(f"Drift levels: {drift_levels} m/s")
    print(f"Seeds per noise combo: {n_seeds}")
    print(f"Total runs: {len(work)} using {n_workers} workers\n")

    # Run
    results = []
    done = 0
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(_run_one, work):
            done += 1
            if r is not None:
                results.append(r)
            if done % 100 == 0 or done == len(work):
                print(f"  [{done}/{len(work)}] completed...")

    # Aggregate: mean over bags and seeds
    from collections import defaultdict
    agg = defaultdict(list)
    for cfg_name, bag_name, noise_m, drift_m_s, seed, med, p90 in results:
        agg[(cfg_name, noise_m, drift_m_s)].append((med, p90))

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  NOISE ROBUSTNESS: mean median error (m) across bags & seeds")
    print(f"{'='*80}")

    # Group by drift level
    for drift_m_s in drift_levels:
        print(f"\n  Drift bias = {drift_m_s} m/s:")
        print(f"  {'noise':>6s}", end="")
        for cfg_name in configs:
            print(f"  {cfg_name:>10s}", end="")
        print(f"  {'delta':>8s}")
        print(f"  {'-'*6}", end="")
        for _ in configs:
            print(f"  {'-'*10}", end="")
        print(f"  {'-'*8}")

        for noise_m in noise_levels:
            print(f"  {noise_m:5.1f}m", end="")
            vals = {}
            for cfg_name in configs:
                key = (cfg_name, noise_m, drift_m_s)
                if key in agg:
                    meds = [x[0] for x in agg[key]]
                    v = np.mean(meds)
                    vals[cfg_name] = v
                    print(f"  {v:9.1f}m", end="")
                else:
                    print(f"  {'N/A':>10s}", end="")
            if len(vals) == 2:
                d = vals.get("tuned", 0) - vals.get("current", 0)
                print(f"  {d:+7.1f}m", end="")
            print()

    # Overall summary
    print(f"\n{'='*80}")
    print(f"  OVERALL MEAN (across all noise+drift combos)")
    print(f"{'='*80}")
    for cfg_name in configs:
        all_meds = []
        for key, vals in agg.items():
            if key[0] == cfg_name:
                all_meds.extend([x[0] for x in vals])
        if all_meds:
            print(f"  {cfg_name:>10s}: mean_median={np.mean(all_meds):.1f}m  "
                  f"mean_p90={np.mean([x[1] for vals in agg.values() for x in vals if True]):.1f}m")


if __name__ == "__main__":
    main()
