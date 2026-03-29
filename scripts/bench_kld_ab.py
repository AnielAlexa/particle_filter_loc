#!/usr/bin/env python3
"""A/B benchmark: particle filter with vs without KLD-adaptive particle count.

Runs all cached bags twice (KLD off, KLD on) and prints a side-by-side
comparison table.

Usage:
  python3 scripts/bench_kld_ab.py
  python3 scripts/bench_kld_ab.py --filter Day2.6 Day3.1
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from replay_mcap import run_replay, load_config


def _merge_config(base_cfg: dict, bag_entry: dict) -> dict:
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


def run_bag(base_cfg, bag_entry, kld_enabled: bool) -> dict:
    cfg = _merge_config(base_cfg, bag_entry)
    cfg["particle_filter"]["kld_enabled"] = kld_enabled
    bag_name = bag_entry["name"]
    cache_file = PKG_DIR / "results" / bag_name / "match_cache.pkl"
    if not cache_file.exists():
        return None
    try:
        return run_replay(cfg, cache_path=str(cache_file), bag_name=bag_name)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="KLD A/B benchmark")
    parser.add_argument("--bags", default=str(PKG_DIR / "config" / "bags.yaml"))
    parser.add_argument("--filter", nargs="+", metavar="BAG")
    parser.add_argument("--rtk-noise", type=float, default=0.0)
    args = parser.parse_args()

    bags_yaml = yaml.safe_load(open(args.bags))
    base_cfg = load_config(str(PKG_DIR / bags_yaml["base_config"]))

    bag_list = bags_yaml["bags"]
    if args.filter:
        bag_list = [b for b in bag_list if b["name"] in args.filter]

    results_off = {}
    results_on = {}

    # --- Run KLD OFF ---
    print("\n" + "=" * 60)
    print("  KLD OFF (fixed particle count)")
    print("=" * 60)
    for bag_entry in bag_list:
        name = bag_entry["name"]
        print(f"  {name} ... ", end="", flush=True)
        r = run_bag(base_cfg, bag_entry, kld_enabled=False)
        if r:
            results_off[name] = r
            print(f"med_trk={r['median_err_tracking']:.1f}m  p90={r['p90_err_tracking']:.1f}m")
        else:
            print("SKIP")

    # --- Run KLD ON ---
    print("\n" + "=" * 60)
    print("  KLD ON (adaptive particle count)")
    print("=" * 60)
    for bag_entry in bag_list:
        name = bag_entry["name"]
        print(f"  {name} ... ", end="", flush=True)
        r = run_bag(base_cfg, bag_entry, kld_enabled=True)
        if r:
            results_on[name] = r
            print(f"med_trk={r['median_err_tracking']:.1f}m  p90={r['p90_err_tracking']:.1f}m")
        else:
            print("SKIP")

    # --- Comparison table ---
    common = sorted(set(results_off) & set(results_on))
    if not common:
        print("\nNo common bags to compare.")
        return

    hdr = (
        f"{'bag':<14} "
        f"{'med_off':>8} {'med_on':>8} {'delta':>7} "
        f"{'p90_off':>8} {'p90_on':>8} {'delta':>7} "
        f"{'conv_off':>8} {'conv_on':>8}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    deltas_med = []
    deltas_p90 = []
    for name in common:
        off = results_off[name]
        on = results_on[name]
        d_med = on["median_err_tracking"] - off["median_err_tracking"]
        d_p90 = on["p90_err_tracking"] - off["p90_err_tracking"]
        deltas_med.append(d_med)
        deltas_p90.append(d_p90)
        print(
            f"{name:<14} "
            f"{off['median_err_tracking']:>7.1f}m {on['median_err_tracking']:>7.1f}m {d_med:>+6.1f}m "
            f"{off['p90_err_tracking']:>7.1f}m {on['p90_err_tracking']:>7.1f}m {d_p90:>+6.1f}m "
            f"{off['converge_s']:>7.1f}s {on['converge_s']:>7.1f}s"
        )

    print(sep)
    print(
        f"{'MEAN':<14} "
        f"{'':>8} {'':>8} {np.mean(deltas_med):>+6.1f}m "
        f"{'':>8} {'':>8} {np.mean(deltas_p90):>+6.1f}m"
    )
    print(sep)
    print(f"\n  + = KLD worse, - = KLD better")


if __name__ == "__main__":
    main()
