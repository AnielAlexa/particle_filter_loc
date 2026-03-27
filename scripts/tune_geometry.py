#!/usr/bin/env python3
"""Sweep geometry weight and RTK noise using cache replay."""

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

from replay_mcap import run_replay, load_config


@contextlib.contextmanager
def suppress_stdout():
    """Suppress print output during parameter sweeps."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def run_one(base_cfg, bag, cache_dir, trust_overrides, pf_overrides=None, rtk_noise=0.0):
    """Run a single bag with overrides, return summary dict."""
    cfg = copy.deepcopy(base_cfg)
    for k, v in trust_overrides.items():
        cfg["trust"][k] = v
    if pf_overrides:
        for k, v in pf_overrides.items():
            cfg["particle_filter"][k] = v

    cfg["replay"]["mcap_path"] = bag["mcap_path"]
    cfg["replay"].pop("output_csv", None)
    cfg["replay"].pop("output_plot", None)
    if "start_offset_s" in bag:
        cfg["replay"]["start_offset_s"] = bag["start_offset_s"]

    cache_path = str(cache_dir / bag["name"] / "match_cache.pkl")
    try:
        with suppress_stdout():
            return run_replay(cfg, cache_path=cache_path, bag_name=bag["name"],
                              rtk_noise_m=rtk_noise)
    except Exception:
        return None


def run_sweep(base_cfg, bags, cache_dir, trust_ov, label, pf_ov=None, rtk_noise=0.0):
    """Run all bags, return aggregated result."""
    meds, p90s, fines = [], [], []
    per_bag = {}
    for bag in bags:
        s = run_one(base_cfg, bag, cache_dir, trust_ov, pf_ov, rtk_noise)
        if s is None or s["n_frames"] == 0:
            continue
        med = s["median_err_tracking"] if s["median_err_tracking"] >= 0 else s["median_err_all"]
        p90 = s.get("p90_err_tracking", s.get("p90_err", -1))
        fine = s.get("fine_rate_pct", 0)
        if med >= 0:
            meds.append(med)
        if p90 >= 0:
            p90s.append(p90)
        fines.append(fine)
        per_bag[bag["name"]] = {"med": med, "p90": p90, "fine": fine}

    return {
        "label": label,
        "mean_med": np.mean(meds) if meds else -1,
        "mean_p90": np.mean(p90s) if p90s else -1,
        "mean_fine": np.mean(fines) if fines else 0,
        "n_bags": len(meds),
        "per_bag": per_bag,
    }


def print_table(results, bag_names):
    bag_cols = "  ".join(f"{n:>8s}" for n in bag_names)
    hdr = f"{'config':<35s} {'med':>6s} {'p90':>6s} {'fine%':>5s} {'bags':>4s}  {bag_cols}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        bv = []
        for n in bag_names:
            b = r["per_bag"].get(n)
            bv.append(f"{b['med']:8.1f}" if b and b["med"] >= 0 else f"{'—':>8s}")
        bs = "  ".join(bv)
        mm = f"{r['mean_med']:.1f}" if r["mean_med"] >= 0 else "—"
        mp = f"{r['mean_p90']:.1f}" if r["mean_p90"] >= 0 else "—"
        print(f"{r['label']:<35s} {mm:>6s} {mp:>6s} {r['mean_fine']:5.1f} {r['n_bags']:>4d}  {bs}")


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

    # ── SWEEP 1: geometry weight ──────────────────────────────────────
    print("=" * 80)
    print("SWEEP 1: geometry_weight (no RTK noise)")
    print("=" * 80)

    configs = [
        ("no_geometry", {
            "geometry_weight": 0.0,
            "inlier_weight": 0.35, "sim_weight": 0.20,
            "consistency_weight": 0.15, "agreement_weight": 0.10,
            "altitude_weight": 0.10, "temporal_weight": 0.10,
        }),
        ("geom=0.05", {
            "geometry_weight": 0.05,
            "inlier_weight": 0.35, "sim_weight": 0.20,
            "consistency_weight": 0.13, "agreement_weight": 0.09,
            "altitude_weight": 0.10, "temporal_weight": 0.08,
        }),
        ("geom=0.10", {
            "geometry_weight": 0.10,
            "inlier_weight": 0.35, "sim_weight": 0.20,
            "consistency_weight": 0.10, "agreement_weight": 0.07,
            "altitude_weight": 0.10, "temporal_weight": 0.08,
        }),
        ("geom=0.15 (current)", {
            "geometry_weight": 0.15,
            "inlier_weight": 0.35, "sim_weight": 0.20,
            "consistency_weight": 0.10, "agreement_weight": 0.05,
            "altitude_weight": 0.10, "temporal_weight": 0.05,
        }),
        ("geom=0.20", {
            "geometry_weight": 0.20,
            "inlier_weight": 0.30, "sim_weight": 0.18,
            "consistency_weight": 0.10, "agreement_weight": 0.05,
            "altitude_weight": 0.10, "temporal_weight": 0.07,
        }),
        ("geom=0.25", {
            "geometry_weight": 0.25,
            "inlier_weight": 0.28, "sim_weight": 0.17,
            "consistency_weight": 0.10, "agreement_weight": 0.05,
            "altitude_weight": 0.10, "temporal_weight": 0.05,
        }),
    ]

    results = []
    for label, tc in configs:
        print(f"  Running {label}...", end="", flush=True)
        r = run_sweep(base_cfg, bags, cache_dir, tc, label)
        mm = f"{r['mean_med']:.1f}m" if r["mean_med"] >= 0 else "—"
        print(f"  med={mm}  bags={r['n_bags']}")
        results.append(r)

    print()
    print_table(results, names)

    # Find best
    valid = [r for r in results if r["mean_med"] >= 0]
    if not valid:
        print("\nNo valid results. Check cache files.")
        return
    best = min(valid, key=lambda r: r["mean_med"])
    best_idx = [r["label"] for r in results].index(best["label"])
    best_trust = dict(configs[best_idx][1])
    print(f"\n>>> Best: {best['label']}  mean_med={best['mean_med']:.1f}m  mean_p90={best['mean_p90']:.1f}m")

    # ── SWEEP 2: RTK noise with best geometry config ──────────────────
    print("\n" + "=" * 80)
    print(f"SWEEP 2: RTK noise stress test ({best['label']})")
    print("=" * 80)

    noise_results = []
    for noise in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        label = f"geom+noise={noise:.1f}m"
        print(f"  Running {label}...", end="", flush=True)
        r = run_sweep(base_cfg, bags, cache_dir, best_trust, label, rtk_noise=noise)
        mm = f"{r['mean_med']:.1f}m" if r["mean_med"] >= 0 else "—"
        print(f"  med={mm}")
        noise_results.append(r)

    print()
    print_table(noise_results, names)

    # ── SWEEP 3: Baseline (no geometry) under noise for comparison ────
    print("\n" + "=" * 80)
    print("SWEEP 3: Baseline (no geometry) under same noise levels")
    print("=" * 80)

    baseline_trust = dict(configs[0][1])
    baseline_results = []
    for noise in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        label = f"base+noise={noise:.1f}m"
        print(f"  Running {label}...", end="", flush=True)
        r = run_sweep(base_cfg, bags, cache_dir, baseline_trust, label, rtk_noise=noise)
        mm = f"{r['mean_med']:.1f}m" if r["mean_med"] >= 0 else "—"
        print(f"  med={mm}")
        baseline_results.append(r)

    # ── Final comparison ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL: Geometry vs Baseline under noise")
    print("=" * 80)
    combined = []
    for r1, r2 in zip(noise_results, baseline_results):
        combined.append(r1)
        combined.append(r2)
    print_table(combined, names)


if __name__ == "__main__":
    main()
