#!/usr/bin/env python3
"""Test adaptive PF: compare clean RTK vs VIO drift across all bags."""

import copy
import csv
import io
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
    label, pf_dict, trust_dict, bag_name, cache_path, lock_cfg, alt_min, drift, seed = args

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
                               f"adap_{label}_{bag_name}_d{drift}_s{seed}.csv")
        _replay_from_cache(
            cache_path=cache_path,
            pf_config=pf_cfg,
            trust_config=t_cfg,
            lock_cfg=lock_cfg,
            altitude_min_m=alt_min,
            output_csv=out_csv,
            output_plot="",
            rtk_noise_m=0.0,
            drift_bias_m_per_s=drift,
        )
        errs = read_csv_errors(out_csv)
        os.unlink(out_csv)
        if errs:
            return (label, bag_name, drift, seed,
                    float(np.median(errs)), float(np.percentile(errs, 90)))
        return None
    except Exception as e:
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
    print(f"Found {len(bags)} bags")

    trust_cfg_dict = copy.deepcopy(base_cfg.get("trust", {}))
    trust_cfg_dict.pop("recon_sim_enabled", None)
    if "coarse_teleport_sigma_range" in trust_cfg_dict:
        v = trust_cfg_dict["coarse_teleport_sigma_range"]
        if isinstance(v, list):
            trust_cfg_dict["coarse_teleport_sigma_range"] = tuple(v)

    lock_cfg = base_cfg.get("init", {})
    alt_min = base_cfg["replay"].get("altitude_min_process_m", 0)

    # Three configs to compare
    configs = {}

    # 1. Adaptive fine_every=2
    adap2_pf = copy.deepcopy(base_cfg["particle_filter"])
    adap2_pf["fine_every_n_frames"] = 2
    configs["adaptive_f2"] = adap2_pf

    # 2. Adaptive fine_every=3
    adap3_pf = copy.deepcopy(base_cfg["particle_filter"])
    adap3_pf["fine_every_n_frames"] = 3
    configs["adaptive_f3"] = adap3_pf

    # 3. Static RTK-optimal
    rtk_pf = copy.deepcopy(base_cfg["particle_filter"])
    rtk_pf["fine_every_n_frames"] = 3
    rtk_pf["adaptive_enabled"] = False
    rtk_pf["sigma_pos_tracking"] = 0.5
    rtk_pf["sigma_obs_fine"] = 15.0
    rtk_pf["roughen_scale"] = 0.5
    configs["static_rtk"] = rtk_pf

    # 4. Static VIO-optimal
    vio_pf = copy.deepcopy(base_cfg["particle_filter"])
    vio_pf["fine_every_n_frames"] = 3
    vio_pf["adaptive_enabled"] = False
    vio_pf["sigma_pos_tracking"] = 3.0
    vio_pf["sigma_obs_fine"] = 5.0
    vio_pf["roughen_scale"] = 2.0
    configs["static_vio"] = vio_pf

    drift_levels = [0.0, 0.5, 1.0, 1.5]
    n_seeds = 3
    n_workers = min(cpu_count(), 4)  # Jetson Orin: leave headroom on 6-core CPU

    work = []
    for label, pf_dict in configs.items():
        for drift in drift_levels:
            seeds = [42] if drift == 0.0 else list(range(n_seeds))
            for seed in seeds:
                for bag_name, cache_path in bags:
                    work.append((label, pf_dict, trust_cfg_dict, bag_name,
                                 cache_path, lock_cfg, alt_min, drift, seed))

    print(f"Total runs: {len(work)} using {n_workers} workers\n")

    from collections import defaultdict
    results = defaultdict(list)
    done = 0
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(_run_one, work):
            done += 1
            if r is not None:
                label, bag_name, drift, seed, med, p90 = r
                results[(label, drift)].append((bag_name, med, p90))
            if done % 50 == 0 or done == len(work):
                print(f"  [{done}/{len(work)}]...")

    # Print summary
    print(f"\n{'='*80}")
    print(f"  ADAPTIVE vs STATIC: mean median error (m)")
    print(f"{'='*80}")
    print(f"  {'drift':>6s}", end="")
    for label in configs:
        print(f"  {label:>12s}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in configs:
        print(f"  {'-'*12}", end="")
    print()

    for drift in drift_levels:
        print(f"  {drift:5.1f}m", end="")
        for label in configs:
            key = (label, drift)
            if key in results:
                meds = [x[1] for x in results[key]]
                print(f"  {np.mean(meds):11.1f}m", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    # Per-bag breakdown for drift=0.0 and drift=1.5
    for drift in [0.0, 1.5]:
        print(f"\n  Per-bag at drift={drift}:")
        print(f"  {'bag':<12s}", end="")
        for label in configs:
            print(f"  {label:>12s}", end="")
        print()
        for bag_name, _ in bags:
            print(f"  {bag_name:<12s}", end="")
            for label in configs:
                key = (label, drift)
                bag_vals = [x[1] for x in results.get(key, []) if x[0] == bag_name]
                if bag_vals:
                    print(f"  {np.mean(bag_vals):11.1f}m", end="")
                else:
                    print(f"  {'N/A':>12s}", end="")
            print()


if __name__ == "__main__":
    main()
