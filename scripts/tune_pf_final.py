#!/usr/bin/env python3
"""Final PF parameter tuning at sub=2, drift=1.5, across noise levels.

Sweeps individual params at noise=0 and noise=2, then combines best,
then validates across all noise levels.
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

SUB = 2


def sweep(base_cfg, bags, cache_dir, ov, label, noise=0.0):
    return run_sweep(base_cfg, bags, cache_dir, ov, label, subsample=SUB, rtk_noise=noise)


def print_table(results, names):
    bag_cols = "  ".join(f"{n:>7s}" for n in names)
    hdr = f"{'config':<42s} {'med':>6s} {'p90':>6s}  {bag_cols}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        bv = []
        for n in names:
            b = r["per_bag"].get(n)
            bv.append(f"{b['med']:7.1f}" if b and b["med"] >= 0 else f"{'—':>7s}")
        bs = "  ".join(bv)
        mm = f"{r['mean_med']:.1f}" if r["mean_med"] >= 0 else "—"
        mp = f"{r['mean_p90']:.1f}" if r["mean_p90"] >= 0 else "—"
        print(f"{r['label']:<42s} {mm:>6s} {mp:>6s}  {bs}")


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
    print(f"Fixed: sub={SUB}, drift=1.5 m/√s\n")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Individual sweeps at noise=0 AND noise=2
    # ══════════════════════════════════════════════════════════════════
    for NOISE in [0.0, 2.0]:
        print(f"\n{'='*80}")
        print(f"PHASE 1: Individual sweeps at noise={NOISE:.0f}m")
        print(f"{'='*80}")

        results = []
        r = sweep(base_cfg, bags, cache_dir, {}, f"baseline", NOISE)
        print(f"  baseline: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        results.append(r)

        # Student-t nu
        for nu in [3.0, 5.0, 10.0, 50.0, 200.0]:
            ov = {"particle_filter": {"likelihood_nu": nu}}
            label = f"nu={nu:.0f}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # Roughening
        for rs in [0.0, 0.3, 0.5, 1.0, 2.0, 3.0]:
            ov = {"particle_filter": {"roughen_enabled": rs > 0, "roughen_scale": rs}}
            label = f"roughen={rs:.1f}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # Process noise
        for sp in [0.3, 0.5, 1.0, 1.5, 2.0]:
            ov = {"particle_filter": {"sigma_pos_tracking": sp}}
            label = f"proc={sp:.1f}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # Fine gate
        for fg in [0.0, 20.0, 30.0, 40.0, 60.0]:
            ov = {"particle_filter": {"fine_consistency_max_m": fg}}
            label = f"gate={fg:.0f}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # Particles
        for nd, nt in [(200, 100), (300, 150), (500, 250)]:
            ov = {"particle_filter": {"n_dispersed": nd, "n_tracking": nt}}
            label = f"part={nd}/{nt}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # Fine sigma
        for sf in [3.0, 5.0, 8.0, 10.0]:
            ov = {"particle_filter": {"sigma_obs_fine": sf}}
            label = f"sig_fine={sf:.0f}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # Coarse sigma
        for sc in [5.0, 10.0, 15.0, 20.0]:
            ov = {"particle_filter": {"sigma_obs_coarse": sc}}
            label = f"sig_coarse={sc:.0f}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # ESS threshold
        for ess in [0.3, 0.5, 0.7]:
            ov = {"particle_filter": {"ess_threshold_fraction": ess}}
            label = f"ess={ess:.1f}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # Sigma modulation
        for smin, smax in [(0.3, 3.0), (0.3, 5.0), (0.5, 5.0), (1.0, 5.0)]:
            ov = {"trust": {"sigma_min_scale": smin, "sigma_max_scale": smax}}
            label = f"sigmod=[{smin},{smax}]"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        # Geometry weight
        for gw in [0.0, 0.10, 0.15, 0.20, 0.25]:
            # Redistribute to keep sum=1
            leftover = 0.15 - gw  # current geom is 0.15
            ov = {"trust": {"geometry_weight": gw,
                            "consistency_weight": 0.10 + leftover * 0.5,
                            "temporal_weight": 0.05 + leftover * 0.5}}
            label = f"geom={gw:.2f}"
            r = sweep(base_cfg, bags, cache_dir, ov, label, NOISE)
            print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
            results.append(r)

        print()
        print_table(results, names)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Combine best individual params
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PHASE 2: Combined configs across noise levels")
    print(f"{'='*80}")

    combos = [
        ("current (baseline)", {}),
        ("roughen2+proc1", {"particle_filter": {
            "roughen_scale": 2.0, "sigma_pos_tracking": 1.0}}),
        ("roughen2+proc1+nu200", {"particle_filter": {
            "roughen_scale": 2.0, "sigma_pos_tracking": 1.0, "likelihood_nu": 200.0}}),
        ("roughen2+proc1+gate20", {"particle_filter": {
            "roughen_scale": 2.0, "sigma_pos_tracking": 1.0, "fine_consistency_max_m": 20.0}}),
        ("roughen2+proc1+sig_fine3", {"particle_filter": {
            "roughen_scale": 2.0, "sigma_pos_tracking": 1.0, "sigma_obs_fine": 3.0}}),
        ("roughen2+proc1+500p", {"particle_filter": {
            "roughen_scale": 2.0, "sigma_pos_tracking": 1.0,
            "n_dispersed": 500, "n_tracking": 250}}),
        ("roughen2+proc1+gate20+sig_f3", {"particle_filter": {
            "roughen_scale": 2.0, "sigma_pos_tracking": 1.0,
            "fine_consistency_max_m": 20.0, "sigma_obs_fine": 3.0}}),
        ("roughen2+proc1+gate20+500p", {"particle_filter": {
            "roughen_scale": 2.0, "sigma_pos_tracking": 1.0,
            "fine_consistency_max_m": 20.0, "n_dispersed": 500, "n_tracking": 250}}),
        ("roughen3+proc1.5", {"particle_filter": {
            "roughen_scale": 3.0, "sigma_pos_tracking": 1.5}}),
        ("roughen2+proc1+geom0.20", {"particle_filter": {
            "roughen_scale": 2.0, "sigma_pos_tracking": 1.0},
            "trust": {"geometry_weight": 0.20, "consistency_weight": 0.075, "temporal_weight": 0.025}}),
    ]

    combo_results = []
    for label, ov in combos:
        noise_meds = []
        for noise in [0.0, 1.0, 2.0, 3.0, 5.0]:
            r = sweep(base_cfg, bags, cache_dir, ov, f"{label}|n={noise:.0f}", noise)
            noise_meds.append((noise, r['mean_med'], r['mean_p90']))
            combo_results.append(r)
        summary = " | ".join(f"n={n:.0f}:{m:.1f}m" for n, m, _ in noise_meds)
        print(f"  {label}: {summary}")

    print()
    print_table(combo_results, names)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Final validation of top 3 combos
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PHASE 3: Top combos - noise sweep summary")
    print(f"{'='*80}")

    # Group by combo label (strip |n=X suffix)
    from collections import defaultdict
    grouped = defaultdict(dict)
    for r in combo_results:
        parts = r["label"].rsplit("|n=", 1)
        if len(parts) == 2:
            combo_label = parts[0]
            noise_str = parts[1]
            grouped[combo_label][noise_str] = r

    print(f"\n{'config':<42s} {'n=0':>7s} {'n=1':>7s} {'n=2':>7s} {'n=3':>7s} {'n=5':>7s}  {'avg':>7s}")
    print("-" * 90)
    for combo_label in [c[0] for c in combos]:
        noise_map = grouped.get(combo_label, {})
        vals = []
        for n in ["0", "1", "2", "3", "5"]:
            r = noise_map.get(n)
            if r and r["mean_med"] >= 0:
                vals.append(r["mean_med"])
            else:
                vals.append(float("nan"))
        avg = np.nanmean(vals)
        val_strs = [f"{v:7.1f}" if not np.isnan(v) else f"{'—':>7s}" for v in vals]
        print(f"{combo_label:<42s} {'  '.join(val_strs)}  {avg:7.1f}")


if __name__ == "__main__":
    main()
