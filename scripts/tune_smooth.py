#!/usr/bin/env python3
"""Tune PF for trajectory smoothness + accuracy at sub=3.

Jitter = RMS of per-frame velocity error (PF vs GT).
Lower jitter = smoother trajectory relative to RTK.
"""

import copy, io, sys, contextlib
from pathlib import Path
import numpy as np, yaml

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(PKG_DIR / "scripts"))

from replay_mcap import run_replay, load_config

SUB = 3  # fixed


@contextlib.contextmanager
def suppress():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try: yield
    finally: sys.stdout = old


def run_one(base_cfg, bag, cache_dir, overrides, rtk_noise=0.0):
    cfg = copy.deepcopy(base_cfg)
    for section, kvs in overrides.items():
        if section in cfg:
            for k, v in kvs.items():
                cfg[section][k] = v
    cfg["replay"]["mcap_path"] = bag["mcap_path"]
    cfg["replay"]["camera_subsample"] = SUB
    cfg["replay"].pop("output_csv", None)
    cfg["replay"].pop("output_plot", None)
    if "start_offset_s" in bag:
        cfg["replay"]["start_offset_s"] = bag["start_offset_s"]
    cache_path = str(cache_dir / bag["name"] / "match_cache.pkl")
    try:
        with suppress():
            return run_replay(cfg, cache_path=cache_path, bag_name=bag["name"],
                              rtk_noise_m=rtk_noise)
    except Exception:
        return None


def run_sweep(base_cfg, bags, cache_dir, overrides, label, rtk_noise=0.0):
    meds, jitters, p90s = [], [], []
    per_bag = {}
    for bag in bags:
        s = run_one(base_cfg, bag, cache_dir, overrides, rtk_noise)
        if s is None or s["n_frames"] == 0:
            continue
        med = s["median_err_tracking"] if s["median_err_tracking"] >= 0 else s["median_err_all"]
        jit = s.get("jitter_m", -1)
        p90 = s.get("p90_err_tracking", s.get("p90_err", -1))
        if med >= 0: meds.append(med)
        if jit >= 0: jitters.append(jit)
        if p90 >= 0: p90s.append(p90)
        per_bag[bag["name"]] = {"med": med, "jit": jit, "p90": p90}
    return {
        "label": label, "n_bags": len(meds),
        "mean_med": np.mean(meds) if meds else -1,
        "mean_jit": np.mean(jitters) if jitters else -1,
        "mean_p90": np.mean(p90s) if p90s else -1,
        "per_bag": per_bag,
    }


def print_table(results, names):
    bag_cols = "  ".join(f"{n:>12s}" for n in names)
    hdr = f"{'config':<38s} {'med':>6s} {'jit':>6s} {'p90':>6s}  {bag_cols}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        bv = []
        for n in names:
            b = r["per_bag"].get(n)
            if b:
                bv.append(f"{b['med']:5.1f}/{b['jit']:.2f}" if b["jit"] >= 0 else f"{b['med']:5.1f}/  —")
            else:
                bv.append(f"{'—':>12s}")
        bs = "  ".join(bv)
        mm = f"{r['mean_med']:.1f}" if r["mean_med"] >= 0 else "—"
        mj = f"{r['mean_jit']:.2f}" if r["mean_jit"] >= 0 else "—"
        mp = f"{r['mean_p90']:.1f}" if r["mean_p90"] >= 0 else "—"
        print(f"{r['label']:<38s} {mm:>6s} {mj:>6s} {mp:>6s}  {bs}")


def main():
    with open(PKG_DIR / "config" / "bags.yaml") as f:
        bags_cfg = yaml.safe_load(f)
    base_cfg = load_config(str(PKG_DIR / bags_cfg["base_config"]))
    cache_dir = PKG_DIR / "results"
    bags = [b for b in bags_cfg["bags"]
            if (cache_dir / b["name"] / "match_cache.pkl").exists()]
    names = [b["name"] for b in bags]
    print(f"Bags: {names}")
    print(f"Format: med/jitter (lower jitter = smoother)\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Measure baseline jitter at noise=0 and noise=2
    # ══════════════════════════════════════════════════════════════
    for NOISE in [0.0, 2.0]:
        print(f"\n{'='*80}")
        print(f"INDIVIDUAL PARAMS at noise={NOISE:.0f}m (sub={SUB})")
        print(f"{'='*80}")

        results = []
        r = run_sweep(base_cfg, bags, cache_dir, {}, "baseline", NOISE)
        print(f"  baseline: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
        results.append(r)

        # Roughening (affects jitter directly)
        for rs in [0.0, 0.3, 0.5, 1.0, 2.0]:
            ov = {"particle_filter": {"roughen_enabled": rs > 0, "roughen_scale": rs}}
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"roughen={rs}", NOISE)
            print(f"  roughen={rs}: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
            results.append(r)

        # Process noise
        for sp in [0.3, 0.5, 1.0, 1.5, 2.0]:
            ov = {"particle_filter": {"sigma_pos_tracking": sp}}
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"proc={sp}", NOISE)
            print(f"  proc={sp}: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
            results.append(r)

        # Drift noise
        for dn in [0.5, 1.0, 1.5, 2.0, 3.0]:
            ov = {"particle_filter": {"drift_noise_m_per_s": dn, "drift_noise_hdg_per_s": dn*1.5}}
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"drift={dn}", NOISE)
            print(f"  drift={dn}: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
            results.append(r)

        # Fine sigma (tighter = more aggressive pulls = more jitter)
        for sf in [3.0, 5.0, 8.0, 12.0, 15.0]:
            ov = {"particle_filter": {"sigma_obs_fine": sf}}
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"sig_fine={sf}", NOISE)
            print(f"  sig_fine={sf}: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
            results.append(r)

        # Coarse sigma
        for sc in [5.0, 10.0, 15.0, 20.0]:
            ov = {"particle_filter": {"sigma_obs_coarse": sc}}
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"sig_coarse={sc}", NOISE)
            print(f"  sig_coarse={sc}: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
            results.append(r)

        # Student-t nu (heavier tails = less aggressive weight changes = smoother?)
        for nu in [3.0, 5.0, 10.0, 50.0, 200.0]:
            ov = {"particle_filter": {"likelihood_nu": nu}}
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"nu={nu:.0f}", NOISE)
            print(f"  nu={nu:.0f}: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
            results.append(r)

        # Sigma modulation range
        for smin, smax in [(0.3, 3.0), (0.3, 5.0), (0.5, 3.0), (1.0, 5.0)]:
            ov = {"trust": {"sigma_min_scale": smin, "sigma_max_scale": smax}}
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"sigmod=[{smin},{smax}]", NOISE)
            print(f"  sigmod=[{smin},{smax}]: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
            results.append(r)

        # ESS threshold (higher = more frequent resampling)
        for ess in [0.3, 0.5, 0.7]:
            ov = {"particle_filter": {"ess_threshold_fraction": ess}}
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"ess={ess}", NOISE)
            print(f"  ess={ess}: med={r['mean_med']:.1f}m  jit={r['mean_jit']:.2f}m")
            results.append(r)

        print()
        print_table(results, names)

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Combined configs optimizing for smoothness
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PHASE 2: Combined smooth configs across noise levels")
    print(f"{'='*80}")

    combos = [
        ("current", {}),
        # Wider fine sigma = gentler correction pulls
        ("sig_fine=10", {"particle_filter": {"sigma_obs_fine": 10.0}}),
        ("sig_fine=12", {"particle_filter": {"sigma_obs_fine": 12.0}}),
        # Lower roughening = less post-resample jitter
        ("roughen=0.5", {"particle_filter": {"roughen_scale": 0.5}}),
        ("roughen=1.0", {"particle_filter": {"roughen_scale": 1.0}}),
        # Combined: wider fine + lower roughen
        ("sig_f10+rough0.5", {"particle_filter": {"sigma_obs_fine": 10.0, "roughen_scale": 0.5}}),
        ("sig_f10+rough1.0", {"particle_filter": {"sigma_obs_fine": 10.0, "roughen_scale": 1.0}}),
        ("sig_f12+rough0.5", {"particle_filter": {"sigma_obs_fine": 12.0, "roughen_scale": 0.5}}),
        # + narrower sigma modulation (less extreme sigma changes)
        ("sig_f10+r0.5+sigmod[0.5,3]", {"particle_filter": {"sigma_obs_fine": 10.0, "roughen_scale": 0.5},
            "trust": {"sigma_min_scale": 0.5, "sigma_max_scale": 3.0}}),
        # + lower drift
        ("sig_f10+r0.5+drift1", {"particle_filter": {"sigma_obs_fine": 10.0, "roughen_scale": 0.5,
            "drift_noise_m_per_s": 1.0, "drift_noise_hdg_per_s": 1.5}}),
        ("sig_f10+r1+proc0.5", {"particle_filter": {"sigma_obs_fine": 10.0, "roughen_scale": 1.0,
            "sigma_pos_tracking": 0.5}}),
    ]

    all_combo = []
    for label, ov in combos:
        row = []
        for noise in [0.0, 1.0, 2.0, 3.0, 5.0]:
            r = run_sweep(base_cfg, bags, cache_dir, ov, f"{label}|n={noise:.0f}", noise)
            row.append(r)
            all_combo.append(r)
        meds = " | ".join(f"n={n:.0f}:{r['mean_med']:.1f}m" for n, r in zip([0,1,2,3,5], row))
        jits = " | ".join(f"{r['mean_jit']:.2f}" for r in row)
        print(f"  {label}")
        print(f"    med:  {meds}")
        print(f"    jit:  {jits}")

    # Final summary: score = median + 5*jitter (penalize jitter)
    print(f"\n{'='*80}")
    print("FINAL: Combined score (median + 5*jitter) — lower is better")
    print(f"{'='*80}")
    by_combo = {}
    for r in all_combo:
        parts = r["label"].rsplit("|n=", 1)
        if len(parts) == 2:
            combo = parts[0]
            if combo not in by_combo:
                by_combo[combo] = []
            by_combo[combo].append(r)

    print(f"{'config':<38s} {'avg_med':>8s} {'avg_jit':>8s} {'score':>8s}")
    print("-" * 65)
    scores = []
    for combo in [c[0] for c in combos]:
        runs = by_combo.get(combo, [])
        if not runs:
            continue
        avg_med = np.mean([r["mean_med"] for r in runs if r["mean_med"] >= 0])
        avg_jit = np.mean([r["mean_jit"] for r in runs if r["mean_jit"] >= 0])
        score = avg_med + 5.0 * avg_jit
        print(f"{combo:<38s} {avg_med:8.1f} {avg_jit:8.2f} {score:8.1f}")
        scores.append((combo, avg_med, avg_jit, score))

    best = min(scores, key=lambda x: x[3])
    print(f"\n>>> Best: {best[0]}  (med={best[1]:.1f}m  jit={best[2]:.2f}m  score={best[3]:.1f})")


if __name__ == "__main__":
    main()
