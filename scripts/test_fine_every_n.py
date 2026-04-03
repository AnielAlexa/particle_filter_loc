#!/usr/bin/env python3
"""A/B test: fine_every_n_frames=2 vs 3 using cache replay."""

import copy
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from scripts.replay_mcap import _replay_from_cache
from particle_filter_loc.particle_filter import PFConfig
from particle_filter_loc.trust_model import TrustConfig


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


def main():
    bags_cfg = load_config(PKG_DIR / "config" / "bags.yaml")
    base_cfg = load_config(PKG_DIR / bags_cfg["base_config"])

    results_dir = PKG_DIR / "results"
    summary = []

    for bag in bags_cfg["bags"]:
        name = bag["name"]
        cache_path = results_dir / name / "match_cache.pkl"
        if not cache_path.exists():
            print(f"[{name}] No cache, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(base_cfg)
        trust_cfg_dict = cfg.get("trust", {})
        trust_cfg_dict.pop("recon_sim_enabled", None)
        if "coarse_teleport_sigma_range" in trust_cfg_dict:
            v = trust_cfg_dict["coarse_teleport_sigma_range"]
            if isinstance(v, list):
                trust_cfg_dict["coarse_teleport_sigma_range"] = tuple(v)
        trust_cfg = TrustConfig(**trust_cfg_dict)

        for n_val in [2, 3]:
            pf_dict = copy.deepcopy(cfg["particle_filter"])
            pf_dict["fine_every_n_frames"] = n_val
            pf_cfg = PFConfig(**pf_dict)

            out_csv = results_dir / name / f"pf_results_fine{n_val}.csv"
            _replay_from_cache(
                cache_path=str(cache_path),
                pf_config=pf_cfg,
                trust_config=trust_cfg,
                lock_cfg=cfg.get("init", {}),
                altitude_min_m=cfg["replay"].get("altitude_min_process_m", 0),
                output_csv=str(out_csv),
                output_plot="",
            )

        err2 = read_csv_errors(str(results_dir / name / "pf_results_fine2.csv"))
        err3 = read_csv_errors(str(results_dir / name / "pf_results_fine3.csv"))

        if err2 and err3:
            med2, p90_2 = np.median(err2), np.percentile(err2, 90)
            med3, p90_3 = np.median(err3), np.percentile(err3, 90)
            summary.append((name, med2, p90_2, med3, p90_3))
            print(f"  every=2: median={med2:.1f}m  p90={p90_2:.1f}m")
            print(f"  every=3: median={med3:.1f}m  p90={p90_3:.1f}m")
            print(f"  delta:   med={med3 - med2:+.1f}m  p90={p90_3 - p90_2:+.1f}m")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: fine_every_n_frames = 2 vs 3")
    print(f"{'='*60}")
    print(f"{'bag':<12s} {'f2_med':>7s} {'f2_p90':>7s} {'f3_med':>8s} {'f3_p90':>8s} {'d_med':>7s} {'d_p90':>7s}")
    print(f"{'-'*12} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
    for name, m2, p2, m3, p3 in summary:
        dm = m3 - m2
        dp = p3 - p2
        tag = "WORSE" if dm > 1.0 else ("better" if dm < -1.0 else "same")
        print(f"{name:<12s} {m2:6.1f}m {p2:6.1f}m  {m3:6.1f}m  {p3:6.1f}m {dm:+6.1f}m {dp:+6.1f}m  {tag}")

    if summary:
        mean2 = np.mean([x[1] for x in summary])
        mean3 = np.mean([x[3] for x in summary])
        print(f"{'MEAN':<12s} {mean2:6.1f}m {'':>7s}  {mean3:6.1f}m {'':>8s} {mean3 - mean2:+6.1f}m")


if __name__ == "__main__":
    main()
