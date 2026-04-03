#!/usr/bin/env python3
"""Tune PF params for VIO regime: 3m relative error in 2s.
Models as random noise + constant drift.
Sweeps sigma_pos, sigma_obs_fine, sigma_pos_dispersed with fine_every=3."""

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
    cfg_idx, pf_dict, trust_dict, bag_name, cache_path, lock_cfg, alt_min, noise_m, drift_m_s, seed = args

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
                               f"vio_{cfg_idx}_{bag_name}_s{seed}.csv")
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
            return (cfg_idx, bag_name, seed,
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
    print(f"Found {len(bags)} bags with caches")

    trust_cfg_dict = copy.deepcopy(base_cfg.get("trust", {}))
    trust_cfg_dict.pop("recon_sim_enabled", None)
    if "coarse_teleport_sigma_range" in trust_cfg_dict:
        v = trust_cfg_dict["coarse_teleport_sigma_range"]
        if isinstance(v, list):
            trust_cfg_dict["coarse_teleport_sigma_range"] = tuple(v)

    lock_cfg = base_cfg.get("init", {})
    alt_min = base_cfg["replay"].get("altitude_min_process_m", 0)

    # VIO drift model: 3m relative error in 2s → 1.5 m/s drift
    # Test pure drift at different rates
    vio_scenarios = [
        ("drift_0.5",  0.0, 0.5),
        ("drift_1.0",  0.0, 1.0),
        ("drift_1.5",  0.0, 1.5),   # matches 3m/2s
        ("drift_2.0",  0.0, 2.0),
    ]

    # Parameter grid
    param_grid = {
        "sigma_pos_tracking":    [0.5, 1.0, 1.5, 2.0, 3.0],
        "sigma_obs_fine":        [5.0, 8.0, 10.0, 15.0, 20.0],
        "roughen_scale":         [0.3, 0.5, 1.0, 2.0],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    n_seeds = 3
    n_workers = min(cpu_count(), 4)  # Jetson Orin: leave headroom on 6-core CPU

    print(f"VIO scenarios: {len(vio_scenarios)}")
    print(f"Param combos: {len(combos)}")
    print(f"Total runs: {len(combos) * len(vio_scenarios) * len(bags) * n_seeds}")
    print(f"Workers: {n_workers}\n")

    # For each VIO scenario, sweep params
    for scenario_name, noise_m, drift_m_s in vio_scenarios:
        print(f"\n{'='*80}")
        print(f"  Scenario: {scenario_name}  (noise={noise_m}m, drift={drift_m_s}m/s)")
        print(f"{'='*80}")

        work = []
        for ci, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            pf_dict = copy.deepcopy(base_cfg["particle_filter"])
            pf_dict["fine_every_n_frames"] = 3
            for k, v in params.items():
                pf_dict[k] = v

            for seed in range(n_seeds):
                for bag_name, cache_path in bags:
                    work.append((ci, pf_dict, trust_cfg_dict, bag_name, cache_path,
                                 lock_cfg, alt_min, noise_m, drift_m_s, seed))

        # Run
        from collections import defaultdict
        combo_results = defaultdict(list)
        done = 0
        with Pool(n_workers) as pool:
            for r in pool.imap_unordered(_run_one, work):
                done += 1
                if r is not None:
                    cfg_idx, bag_name, seed, med, p90 = r
                    combo_results[cfg_idx].append((med, p90))
                if done % 200 == 0 or done == len(work):
                    print(f"  [{done}/{len(work)}]...")

        # Rank
        ranked = []
        for ci, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            vals = combo_results.get(ci, [])
            if vals:
                mean_med = np.mean([v[0] for v in vals])
                mean_p90 = np.mean([v[1] for v in vals])
                ranked.append((params, mean_med, mean_p90))
        ranked.sort(key=lambda x: x[1])

        print(f"\n  TOP 10 for {scenario_name}:")
        print(f"  {'rank':>4s}  {'sig_pos':>7s} {'sig_fine':>8s} {'rough':>6s}  {'mean_med':>8s} {'mean_p90':>8s}")
        print(f"  {'-'*4}  {'-'*7} {'-'*8} {'-'*6}  {'-'*8} {'-'*8}")
        for i, (p, mm, mp) in enumerate(ranked[:10]):
            print(f"  {i+1:4d}  {p['sigma_pos_tracking']:7.1f} {p['sigma_obs_fine']:8.1f} "
                  f"{p['roughen_scale']:6.1f}  {mm:7.1f}m {mp:7.1f}m")

        if ranked:
            # Also show current config performance
            for p, mm, mp in ranked:
                if (p["sigma_pos_tracking"] == 1.0 and
                    p["sigma_obs_fine"] == 10.0 and
                    p["roughen_scale"] == 0.5):
                    idx = ranked.index((p, mm, mp)) + 1
                    print(f"\n  Current config rank: #{idx}/{len(ranked)}  "
                          f"mean_med={mm:.1f}m  mean_p90={mp:.1f}m")
                    break


if __name__ == "__main__":
    main()
