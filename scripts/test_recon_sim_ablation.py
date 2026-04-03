#!/usr/bin/env python3
"""A/B test: recon_sim ON vs OFF (zeroed) using cache replay."""

import copy
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from scripts.replay_mcap import run_replay, _replay_from_cache
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

    bags = bags_cfg["bags"]
    summary = []

    for bag in bags:
        name = bag["name"]
        cache_path = results_dir / name / "match_cache.pkl"
        if not cache_path.exists():
            print(f"[{name}] No cache, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Build configs
        cfg = copy.deepcopy(base_cfg)
        # Apply bag overrides
        for k, v in bag.items():
            if k in ("name", "notes"):
                continue
            if k in cfg.get("replay", {}):
                cfg["replay"][k] = v

        pf_cfg = PFConfig(**cfg["particle_filter"])
        trust_cfg_dict = cfg.get("trust", {})
        if "coarse_teleport_sigma_range" in trust_cfg_dict:
            v = trust_cfg_dict["coarse_teleport_sigma_range"]
            if isinstance(v, list):
                trust_cfg_dict["coarse_teleport_sigma_range"] = tuple(v)

        # --- A) Normal (recon_sim from cache) ---
        trust_a = TrustConfig(**trust_cfg_dict)
        out_a = results_dir / name / "pf_results_recon_ON.csv"
        res_a = _replay_from_cache(
            cache_path=str(cache_path),
            pf_config=pf_cfg,
            trust_config=trust_a,
            lock_cfg=cfg.get("init", {}),
            altitude_min_m=cfg["replay"].get("altitude_min_process_m", 0),
            output_csv=str(out_a),
            output_plot="",
        )

        # --- B) recon_sim disabled: set drift thresholds so drift never fires ---
        trust_b_dict = copy.deepcopy(trust_cfg_dict)
        # Set recon_sim threshold to 0 so drift_detected is never True
        trust_b_dict["drift_recon_sim_threshold"] = 0.0
        trust_b_dict["drift_pf_confidence_threshold"] = 0.0
        trust_b = TrustConfig(**trust_b_dict)
        out_b = results_dir / name / "pf_results_recon_OFF.csv"
        res_b = _replay_from_cache(
            cache_path=str(cache_path),
            pf_config=pf_cfg,
            trust_config=trust_b,
            lock_cfg=cfg.get("init", {}),
            altitude_min_m=cfg["replay"].get("altitude_min_process_m", 0),
            output_csv=str(out_b),
            output_plot="",
        )

        err_a = read_csv_errors(str(out_a))
        err_b = read_csv_errors(str(out_b))

        if err_a and err_b:
            med_a = np.median(err_a)
            p90_a = np.percentile(err_a, 90)
            med_b = np.median(err_b)
            p90_b = np.percentile(err_b, 90)
            summary.append((name, med_a, p90_a, med_b, p90_b))
            print(f"  ON:  median={med_a:.1f}m  p90={p90_a:.1f}m")
            print(f"  OFF: median={med_b:.1f}m  p90={p90_b:.1f}m")
            print(f"  delta: med={med_b - med_a:+.1f}m  p90={p90_b - p90_a:+.1f}m")
        else:
            print(f"  [WARN] No TRACKING errors for {name}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: recon_sim drift detection ON vs OFF")
    print(f"{'='*60}")
    print(f"{'bag':<12s} {'ON_med':>7s} {'ON_p90':>7s} {'OFF_med':>8s} {'OFF_p90':>8s} {'d_med':>7s} {'d_p90':>7s}")
    print(f"{'-'*12} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
    for name, ma, pa, mb, pb in summary:
        dm = mb - ma
        dp = pb - pa
        print(f"{name:<12s} {ma:6.1f}m {pa:6.1f}m  {mb:6.1f}m  {pb:6.1f}m {dm:+6.1f}m {dp:+6.1f}m")

    if summary:
        mean_on = np.mean([x[1] for x in summary])
        mean_off = np.mean([x[3] for x in summary])
        print(f"{'MEAN':<12s} {mean_on:6.1f}m {'':>7s}  {mean_off:6.1f}m {'':>8s} {mean_off - mean_on:+6.1f}m")


if __name__ == "__main__":
    main()
