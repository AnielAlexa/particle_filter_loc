#!/usr/bin/env python3
"""Sweep KLD parameters to find best configuration."""

import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from replay_mcap import run_replay, load_config


def _merge_config(base_cfg, bag_entry):
    cfg = copy.deepcopy(base_cfg)
    replay_keys = {
        "mcap_path", "start_offset_s", "camera_subsample",
        "altitude_min_process_m", "output_csv", "output_plot",
        "camera_topic", "rtk_topic", "yaw_topic", "altimeter_topic",
    }
    pf_keys = {
        "sigma_obs_coarse", "sigma_obs_fine", "init_altitude_m",
        "converge_spread_m", "tracking_spread_m", "lost_spread_m",
        "n_dispersed", "n_tracking", "fine_every_n_frames",
    }
    for k, v in bag_entry.items():
        if k in ("name", "notes"):
            continue
        if k in replay_keys:
            cfg["replay"][k] = v
        elif k in pf_keys:
            cfg["particle_filter"][k] = v
        elif k == "enu_origin":
            cfg["enu_origin"] = v
    cfg["replay"]["mcap_path"] = bag_entry["mcap_path"]
    cfg["replay"].pop("output_csv", None)
    cfg["replay"].pop("output_plot", None)
    return cfg


GRID = {
    "kld_n_min":       [50, 100, 150, 200],
    "kld_n_max":       [300, 500, 700],
    "kld_bin_pos_m":   [3.0, 5.0, 8.0],
    "kld_bin_hdg_deg": [10.0, 15.0, 30.0],
    "kld_epsilon":     [0.03, 0.05, 0.1],
}


def main():
    bags_yaml = yaml.safe_load(open(PKG_DIR / "config" / "bags.yaml"))
    base_cfg = load_config(str(PKG_DIR / bags_yaml["base_config"]))
    bag_list = bags_yaml["bags"]

    # Use a representative subset for speed
    use_bags = ["Day2.1", "Day2.6", "Day3.1", "Day3.2", "Day4.2"]
    bag_list = [b for b in bag_list if b["name"] in use_bags]

    # Verify caches exist
    for b in bag_list:
        cache = PKG_DIR / "results" / b["name"] / "match_cache.pkl"
        if not cache.exists():
            print(f"Missing cache: {b['name']}")
            bag_list = [x for x in bag_list if x["name"] != b["name"]]

    # --- Baseline (KLD off) ---
    print("Running baseline (KLD off)...")
    baseline = {}
    for bag_entry in bag_list:
        name = bag_entry["name"]
        cfg = _merge_config(base_cfg, bag_entry)
        cfg["particle_filter"]["kld_enabled"] = False
        cache = str(PKG_DIR / "results" / name / "match_cache.pkl")
        r = run_replay(cfg, cache_path=cache, bag_name=name)
        baseline[name] = r
        print(f"  {name}: med={r['median_err_tracking']:.1f}m p90={r['p90_err_tracking']:.1f}m")

    base_med = np.mean([baseline[b["name"]]["median_err_tracking"] for b in bag_list])
    base_p90 = np.mean([baseline[b["name"]]["p90_err_tracking"] for b in bag_list])
    print(f"\nBaseline MEAN: med={base_med:.2f}m  p90={base_p90:.2f}m\n")

    # --- Grid sweep ---
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Sweeping {len(combos)} KLD configs across {len(bag_list)} bags...\n")

    results = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        # Skip nonsensical: n_min > n_max
        if params["kld_n_min"] >= params["kld_n_max"]:
            continue

        meds = []
        p90s = []
        ok = True
        for bag_entry in bag_list:
            name = bag_entry["name"]
            cfg = _merge_config(base_cfg, bag_entry)
            cfg["particle_filter"]["kld_enabled"] = True
            for k, v in params.items():
                cfg["particle_filter"][k] = v
            cache = str(PKG_DIR / "results" / name / "match_cache.pkl")
            try:
                r = run_replay(cfg, cache_path=cache, bag_name=name)
                meds.append(r["median_err_tracking"])
                p90s.append(r["p90_err_tracking"])
            except Exception:
                ok = False
                break

        if not ok:
            continue

        avg_med = np.mean(meds)
        avg_p90 = np.mean(p90s)
        d_med = avg_med - base_med
        d_p90 = avg_p90 - base_p90

        results.append({**params, "avg_med": avg_med, "avg_p90": avg_p90,
                        "d_med": d_med, "d_p90": d_p90})

        tag = "***" if d_med < 0 and d_p90 < 0 else ""
        print(f"[{i+1:3d}/{len(combos)}] n_min={params['kld_n_min']:3d} "
              f"n_max={params['kld_n_max']:3d} bin={params['kld_bin_pos_m']:.0f}m/"
              f"{params['kld_bin_hdg_deg']:.0f}° eps={params['kld_epsilon']:.2f} "
              f"| med={avg_med:.2f}m ({d_med:+.2f}) p90={avg_p90:.2f}m ({d_p90:+.2f}) {tag}")

    # --- Top 10 ---
    results.sort(key=lambda r: r["avg_med"] + 0.5 * r["avg_p90"])
    print(f"\n{'='*80}")
    print(f"TOP 10 (sorted by med + 0.5*p90)  |  Baseline: med={base_med:.2f}m p90={base_p90:.2f}m")
    print(f"{'='*80}")
    for j, r in enumerate(results[:10]):
        print(f"  #{j+1}: n_min={r['kld_n_min']:3d} n_max={r['kld_n_max']:3d} "
              f"bin={r['kld_bin_pos_m']:.0f}m/{r['kld_bin_hdg_deg']:.0f}° "
              f"eps={r['kld_epsilon']:.2f} "
              f"| med={r['avg_med']:.2f}m ({r['d_med']:+.2f}) "
              f"p90={r['avg_p90']:.2f}m ({r['d_p90']:+.2f})")


if __name__ == "__main__":
    main()
