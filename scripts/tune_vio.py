#!/usr/bin/env python3
"""Sweep drift_noise vs rtk_noise vs camera_subsample for VIO readiness.

Strategy:
  1. For each subsample rate, find the best drift_noise at each rtk_noise level
  2. Show the full matrix: subsample x rtk_noise x drift_noise
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

from tune_subsample import run_sweep, load_config


def main():
    bags_yaml = PKG_DIR / "config" / "bags.yaml"
    with open(bags_yaml) as f:
        bags_cfg = yaml.safe_load(f)
    base_cfg = load_config(str(PKG_DIR / bags_cfg["base_config"]))
    cache_dir = PKG_DIR / "results"
    bags = [b for b in bags_cfg["bags"]
            if (cache_dir / b["name"] / "match_cache.pkl").exists()]
    names = [b["name"] for b in bags]
    print(f"Bags: {names}\n")

    subsample_rates = [1, 2, 4, 5]
    rtk_noises = [0.0, 1.0, 2.0, 3.0, 5.0]
    drift_noises = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]

    # ══════════════════════════════════════════════════════════════════
    # Full 3D sweep: subsample x rtk_noise x drift_noise
    # ══════════════════════════════════════════════════════════════════
    all_results = {}  # (sub, rtk, drift) -> result

    for sub in subsample_rates:
        print(f"{'='*80}")
        print(f"SUBSAMPLE = {sub}  ({'%d' % (20/sub)}Hz at 20Hz camera)")
        print(f"{'='*80}")

        for rtk in rtk_noises:
            print(f"\n  rtk_noise={rtk:.1f}m:")
            for drift in drift_noises:
                hdg_drift = drift * 1.5
                ov = {"particle_filter": {
                    "drift_noise_m_per_s": drift,
                    "drift_noise_hdg_per_s": hdg_drift,
                }}
                label = f"s{sub}_r{rtk:.0f}_d{drift:.1f}"
                r = run_sweep(base_cfg, bags, cache_dir, ov, label,
                              subsample=sub, rtk_noise=rtk)
                all_results[(sub, rtk, drift)] = r
                med = f"{r['mean_med']:.1f}" if r['mean_med'] >= 0 else "—"
                p90 = f"{r['mean_p90']:.1f}" if r['mean_p90'] >= 0 else "—"
                marker = ""
                print(f"    drift={drift:.1f}: med={med:>6s}m  p90={p90:>6s}m{marker}")

    # ══════════════════════════════════════════════════════════════════
    # Summary: Best drift_noise for each (subsample, rtk_noise) combo
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("BEST drift_noise for each (subsample, rtk_noise)")
    print(f"{'='*80}")

    # Header
    rtk_hdr = "  ".join(f"{'n='+str(int(r))+'m':>18s}" for r in rtk_noises)
    print(f"{'sub':>4s}  {rtk_hdr}")
    print("-" * (6 + len(rtk_hdr)))

    best_configs = {}
    for sub in subsample_rates:
        row = []
        for rtk in rtk_noises:
            candidates = [(drift, all_results[(sub, rtk, drift)])
                          for drift in drift_noises
                          if all_results[(sub, rtk, drift)]["mean_med"] >= 0]
            if candidates:
                best_drift, best_r = min(candidates, key=lambda x: x[1]["mean_med"])
                row.append(f"d={best_drift:.1f} {best_r['mean_med']:.1f}m")
                best_configs[(sub, rtk)] = (best_drift, best_r)
            else:
                row.append("—")
        print(f"  {sub:>2d}  {'  '.join(f'{v:>18s}' for v in row)}")

    # ══════════════════════════════════════════════════════════════════
    # Heatmap: median error for each (subsample, rtk_noise) at best drift
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("MEDIAN ERROR (best drift per cell)")
    print(f"{'='*80}")

    rtk_hdr = "  ".join(f"{'n='+str(int(r))+'m':>8s}" for r in rtk_noises)
    print(f"{'sub':>4s}  {rtk_hdr}")
    print("-" * (6 + len(rtk_hdr)))

    for sub in subsample_rates:
        row = []
        for rtk in rtk_noises:
            if (sub, rtk) in best_configs:
                _, r = best_configs[(sub, rtk)]
                row.append(f"{r['mean_med']:.1f}m")
            else:
                row.append("—")
        print(f"  {sub:>2d}  {'  '.join(f'{v:>8s}' for v in row)}")

    # ══════════════════════════════════════════════════════════════════
    # Per-bag breakdown for key configs
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PER-BAG: Key configs (sub x noise, best drift)")
    print(f"{'='*80}")

    bag_hdr = "  ".join(f"{n:>7s}" for n in names)
    print(f"{'config':<30s} {'med':>6s} {'p90':>6s}  {bag_hdr}")
    print("-" * (30 + 6 + 6 + 4 + len(bag_hdr)))

    key_combos = [(s, n) for s in subsample_rates for n in [0.0, 2.0, 3.0, 5.0]]
    for sub, rtk in key_combos:
        if (sub, rtk) not in best_configs:
            continue
        best_drift, r = best_configs[(sub, rtk)]
        bv = []
        for n in names:
            b = r["per_bag"].get(n)
            bv.append(f"{b['med']:7.1f}" if b and b["med"] >= 0 else f"{'—':>7s}")
        bs = "  ".join(bv)
        label = f"s{sub} n{rtk:.0f}m d{best_drift:.1f}"
        mm = f"{r['mean_med']:.1f}" if r["mean_med"] >= 0 else "—"
        mp = f"{r['mean_p90']:.1f}" if r["mean_p90"] >= 0 else "—"
        print(f"{label:<30s} {mm:>6s} {mp:>6s}  {bs}")

    # ══════════════════════════════════════════════════════════════════
    # Recommendation
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    for sub in subsample_rates:
        print(f"\n  subsample={sub} ({20//sub}Hz):")
        for rtk in rtk_noises:
            if (sub, rtk) in best_configs:
                best_drift, r = best_configs[(sub, rtk)]
                print(f"    rtk_noise={rtk:.0f}m → drift_noise={best_drift:.1f} m/√s  "
                      f"(median={r['mean_med']:.1f}m, p90={r['mean_p90']:.1f}m)")


if __name__ == "__main__":
    main()
