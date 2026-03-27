#!/usr/bin/env python3
"""Multi-parameter PF robustness sweep using cache replay.

Strategy:
  1. Sweep individual parameters at noise=2m to find what helps
  2. Combine best individual changes
  3. Validate combined config at noise=0,1,2,3,5m vs baseline
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

from replay_mcap import run_replay, load_config


@contextlib.contextmanager
def suppress():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def run_sweep(base_cfg, bags, cache_dir, overrides, label, rtk_noise=0.0):
    meds, p90s, fines = [], [], []
    per_bag = {}
    for bag in bags:
        cfg = copy.deepcopy(base_cfg)
        for section, kvs in overrides.items():
            if section in cfg:
                for k, v in kvs.items():
                    cfg[section][k] = v
        cfg["replay"]["mcap_path"] = bag["mcap_path"]
        cfg["replay"].pop("output_csv", None)
        cfg["replay"].pop("output_plot", None)
        if "start_offset_s" in bag:
            cfg["replay"]["start_offset_s"] = bag["start_offset_s"]

        cache_path = str(cache_dir / bag["name"] / "match_cache.pkl")
        try:
            with suppress():
                s = run_replay(cfg, cache_path=cache_path, bag_name=bag["name"],
                               rtk_noise_m=rtk_noise)
        except Exception:
            continue
        if s["n_frames"] == 0:
            continue
        med = s["median_err_tracking"] if s["median_err_tracking"] >= 0 else s["median_err_all"]
        p90 = s.get("p90_err_tracking", s.get("p90_err", -1))
        fine = s.get("fine_rate_pct", 0)
        if med >= 0: meds.append(med)
        if p90 >= 0: p90s.append(p90)
        fines.append(fine)
        per_bag[bag["name"]] = {"med": med, "p90": p90, "fine": fine}

    return {
        "label": label, "n_bags": len(meds),
        "mean_med": np.mean(meds) if meds else -1,
        "mean_p90": np.mean(p90s) if p90s else -1,
        "mean_fine": np.mean(fines) if fines else 0,
        "per_bag": per_bag,
    }


def print_table(results, bag_names):
    bag_cols = "  ".join(f"{n:>7s}" for n in bag_names)
    hdr = f"{'config':<40s} {'med':>6s} {'p90':>6s} {'fine%':>5s}  {bag_cols}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        bv = []
        for n in bag_names:
            b = r["per_bag"].get(n)
            bv.append(f"{b['med']:7.1f}" if b and b["med"] >= 0 else f"{'—':>7s}")
        bs = "  ".join(bv)
        mm = f"{r['mean_med']:.1f}" if r["mean_med"] >= 0 else "—"
        mp = f"{r['mean_p90']:.1f}" if r["mean_p90"] >= 0 else "—"
        print(f"{r['label']:<40s} {mm:>6s} {mp:>6s} {r['mean_fine']:5.1f}  {bs}")


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

    NOISE = 2.0  # Target noise level for optimization

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Individual parameter sweeps at noise=2m
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"PHASE 1: Individual parameter sweeps at RTK noise={NOISE}m")
    print(f"{'='*80}")

    # Current config as baseline
    baseline = {}  # empty = use config as-is
    print("\n--- Baseline (current config) ---")
    r_base = run_sweep(base_cfg, bags, cache_dir, baseline, "baseline", rtk_noise=NOISE)
    print(f"  med={r_base['mean_med']:.1f}m  p90={r_base['mean_p90']:.1f}m")

    all_results = [r_base]

    # 1a. Student-t nu
    print("\n--- Student-t nu ---")
    for nu in [3.0, 5.0, 10.0, 20.0, 200.0]:
        label = f"nu={nu:.0f}" + (" (Gaussian)" if nu >= 100 else "")
        r = run_sweep(base_cfg, bags, cache_dir,
                      {"particle_filter": {"likelihood_nu": nu}}, label, NOISE)
        print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        all_results.append(r)

    # 1b. Roughening scale
    print("\n--- Roughening ---")
    for rs in [0.0, 0.3, 0.5, 1.0, 2.0]:
        label = f"roughen={rs:.1f}" + (" (off)" if rs == 0 else "")
        enabled = rs > 0
        r = run_sweep(base_cfg, bags, cache_dir,
                      {"particle_filter": {"roughen_enabled": enabled, "roughen_scale": rs}},
                      label, NOISE)
        print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        all_results.append(r)

    # 1c. Drift noise
    print("\n--- Drift noise ---")
    for dn in [0.0, 0.3, 0.5, 1.0, 2.0]:
        label = f"drift={dn:.1f}m/√s"
        r = run_sweep(base_cfg, bags, cache_dir,
                      {"particle_filter": {"drift_noise_m_per_s": dn,
                                           "drift_noise_hdg_per_s": dn * 1.5}},
                      label, NOISE)
        print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        all_results.append(r)

    # 1d. Fine consistency gate
    print("\n--- Fine consistency gate ---")
    for fc in [0.0, 20.0, 40.0, 60.0, 100.0]:
        label = f"fine_gate={fc:.0f}m" + (" (off)" if fc == 0 else "")
        r = run_sweep(base_cfg, bags, cache_dir,
                      {"particle_filter": {"fine_consistency_max_m": fc}}, label, NOISE)
        print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        all_results.append(r)

    # 1e. Particle count
    print("\n--- Particle count ---")
    for nd, nt in [(200, 100), (300, 150), (500, 200)]:
        label = f"particles={nd}/{nt}"
        r = run_sweep(base_cfg, bags, cache_dir,
                      {"particle_filter": {"n_dispersed": nd, "n_tracking": nt}},
                      label, NOISE)
        print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        all_results.append(r)

    # 1f. Sigma obs fine
    print("\n--- Fine observation sigma ---")
    for sf in [3.0, 5.0, 8.0, 12.0]:
        label = f"sigma_fine={sf:.0f}m"
        r = run_sweep(base_cfg, bags, cache_dir,
                      {"particle_filter": {"sigma_obs_fine": sf}}, label, NOISE)
        print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        all_results.append(r)

    # 1g. Process noise (tracking)
    print("\n--- Process noise (tracking) ---")
    for sp in [0.3, 0.5, 1.0, 2.0]:
        label = f"sigma_pos_track={sp:.1f}m"
        r = run_sweep(base_cfg, bags, cache_dir,
                      {"particle_filter": {"sigma_pos_tracking": sp}}, label, NOISE)
        print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        all_results.append(r)

    # 1h. Sigma modulation range
    print("\n--- Sigma modulation ---")
    for smin, smax in [(0.5, 3.0), (0.3, 5.0), (1.0, 3.0), (0.5, 5.0)]:
        label = f"sigma_scale=[{smin},{smax}]"
        r = run_sweep(base_cfg, bags, cache_dir,
                      {"trust": {"sigma_min_scale": smin, "sigma_max_scale": smax}},
                      label, NOISE)
        print(f"  {label}: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
        all_results.append(r)

    print(f"\n{'='*80}")
    print("PHASE 1 RESULTS")
    print(f"{'='*80}")
    print_table(all_results, names)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Combine best changes
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PHASE 2: Combined configs at noise=2m")
    print(f"{'='*80}")

    combined_results = [r_base]

    # Combo A: Student-t + roughening
    combo_a = {
        "particle_filter": {"likelihood_nu": 5.0, "roughen_enabled": True, "roughen_scale": 0.5},
    }
    r = run_sweep(base_cfg, bags, cache_dir, combo_a, "combo_A: nu5+roughen0.5", NOISE)
    print(f"  combo_A: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
    combined_results.append(r)

    # Combo B: A + wider consistency gate
    combo_b = {
        "particle_filter": {"likelihood_nu": 5.0, "roughen_enabled": True, "roughen_scale": 0.5,
                            "fine_consistency_max_m": 40.0},
    }
    r = run_sweep(base_cfg, bags, cache_dir, combo_b, "combo_B: A+gate40", NOISE)
    print(f"  combo_B: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
    combined_results.append(r)

    # Combo C: B + more particles
    combo_c = {
        "particle_filter": {"likelihood_nu": 5.0, "roughen_enabled": True, "roughen_scale": 0.5,
                            "fine_consistency_max_m": 40.0,
                            "n_dispersed": 300, "n_tracking": 150},
    }
    r = run_sweep(base_cfg, bags, cache_dir, combo_c, "combo_C: B+300particles", NOISE)
    print(f"  combo_C: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
    combined_results.append(r)

    # Combo D: C + drift noise
    combo_d = {
        "particle_filter": {"likelihood_nu": 5.0, "roughen_enabled": True, "roughen_scale": 0.5,
                            "fine_consistency_max_m": 40.0,
                            "n_dispersed": 300, "n_tracking": 150,
                            "drift_noise_m_per_s": 0.5, "drift_noise_hdg_per_s": 0.75},
    }
    r = run_sweep(base_cfg, bags, cache_dir, combo_d, "combo_D: C+drift0.5", NOISE)
    print(f"  combo_D: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
    combined_results.append(r)

    # Combo E: D + wider sigma modulation
    combo_e = {
        "particle_filter": {"likelihood_nu": 5.0, "roughen_enabled": True, "roughen_scale": 0.5,
                            "fine_consistency_max_m": 40.0,
                            "n_dispersed": 300, "n_tracking": 150,
                            "drift_noise_m_per_s": 0.5, "drift_noise_hdg_per_s": 0.75},
        "trust": {"sigma_min_scale": 0.3, "sigma_max_scale": 5.0},
    }
    r = run_sweep(base_cfg, bags, cache_dir, combo_e, "combo_E: D+sigma[0.3,5]", NOISE)
    print(f"  combo_E: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
    combined_results.append(r)

    # Combo F: Best from above + higher process noise
    combo_f = {
        "particle_filter": {"likelihood_nu": 5.0, "roughen_enabled": True, "roughen_scale": 0.5,
                            "fine_consistency_max_m": 40.0,
                            "n_dispersed": 300, "n_tracking": 150,
                            "drift_noise_m_per_s": 1.0, "drift_noise_hdg_per_s": 1.5,
                            "sigma_pos_tracking": 0.5},
        "trust": {"sigma_min_scale": 0.3, "sigma_max_scale": 5.0},
    }
    r = run_sweep(base_cfg, bags, cache_dir, combo_f, "combo_F: E+drift1.0+proc0.5", NOISE)
    print(f"  combo_F: med={r['mean_med']:.1f}m  p90={r['mean_p90']:.1f}m")
    combined_results.append(r)

    print()
    print_table(combined_results, names)

    # Find best combo
    valid_c = [r for r in combined_results if r["mean_med"] >= 0]
    best_combo = min(valid_c, key=lambda r: r["mean_med"])
    print(f"\n>>> Best combo: {best_combo['label']}  med={best_combo['mean_med']:.1f}m  p90={best_combo['mean_p90']:.1f}m")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Validate best combo across noise levels
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PHASE 3: Best combo vs baseline across noise levels")
    print(f"{'='*80}")

    # Determine which combo was best and get its overrides
    combo_map = {
        "combo_A: nu5+roughen0.5": combo_a,
        "combo_B: A+gate40": combo_b,
        "combo_C: B+300particles": combo_c,
        "combo_D: C+drift0.5": combo_d,
        "combo_E: D+sigma[0.3,5]": combo_e,
        "combo_F: E+drift1.0+proc0.5": combo_f,
    }
    best_ov = combo_map.get(best_combo["label"], {})

    final_results = []
    for noise in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        label_best = f"BEST+noise={noise:.1f}m"
        label_base = f"BASE+noise={noise:.1f}m"
        print(f"  Running noise={noise:.1f}m...", end="", flush=True)
        r_best = run_sweep(base_cfg, bags, cache_dir, best_ov, label_best, noise)
        r_base = run_sweep(base_cfg, bags, cache_dir, {}, label_base, noise)
        delta = r_best["mean_med"] - r_base["mean_med"] if r_best["mean_med"] >= 0 and r_base["mean_med"] >= 0 else 0
        print(f"  best={r_best['mean_med']:.1f}m  base={r_base['mean_med']:.1f}m  delta={delta:+.1f}m")
        final_results.append(r_best)
        final_results.append(r_base)

    print()
    print_table(final_results, names)


if __name__ == "__main__":
    main()
