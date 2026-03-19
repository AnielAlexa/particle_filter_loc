#!/usr/bin/env python3
"""Multi-bag particle filter benchmarking harness.

Runs replay_mcap.py logic on all bags defined in bags.yaml and produces a
summary comparison table + CSV.

Usage:
  # Full TRT replay, generate caches:
  python3 bench_all_bags.py --bags config/bags.yaml --save-cache

  # Re-run PF-only from existing caches (fast, no TRT):
  python3 bench_all_bags.py --bags config/bags.yaml --use-cache

  # Run only specific bags:
  python3 bench_all_bags.py --bags config/bags.yaml --filter Day2.6 Day3.1 --use-cache

  # Debug: limit frames per bag:
  python3 bench_all_bags.py --bags config/bags.yaml --max-frames 200 --save-cache
"""

import argparse
import copy
import csv
import sys
from pathlib import Path

import yaml
import numpy as np

PKG_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from replay_mcap import run_replay, load_config


def _merge_config(base_cfg: dict, bag_entry: dict) -> dict:
    """Deep-copy base config and apply per-bag overrides to the replay section."""
    cfg = copy.deepcopy(base_cfg)
    # Keys that belong to the replay section
    replay_keys = {
        "mcap_path", "start_offset_s", "camera_subsample",
        "altitude_min_process_m", "output_csv", "output_plot",
        "camera_topic", "rtk_topic", "yaw_topic", "altimeter_topic",
    }
    # Keys that go into particle_filter section
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
        # enu_origin override
        elif k == "enu_origin":
            cfg["enu_origin"] = v

    # Always set mcap_path from bag entry
    cfg["replay"]["mcap_path"] = bag_entry["mcap_path"]
    # Remove legacy single-bag output paths so per-bag dirs are used
    cfg["replay"].pop("output_csv", None)
    cfg["replay"].pop("output_plot", None)

    return cfg


def _print_table(rows: list):
    header = (
        f"{'bag':<14} {'n_frames':>8} {'med_all':>8} {'med_trk':>8} "
        f"{'p90_trk':>8} {'conv_s':>7} {'fine%':>6}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['bag']:<14} {r['n_frames']:>8d} "
            f"{r['median_err_all']:>7.1f}m "
            f"{r['median_err_tracking']:>7.1f}m "
            f"{r['p90_err_tracking']:>7.1f}m "
            f"{r['converge_s']:>6.1f}s "
            f"{r['fine_rate_pct']:>5.1f}%"
        )
    print(sep)
    # Aggregate across bags with valid data
    valid = [r for r in rows if r["n_frames"] > 0 and r["median_err_tracking"] >= 0]
    if valid:
        print(
            f"{'MEAN':<14} {'':>8} "
            f"{np.mean([r['median_err_all'] for r in valid]):>7.1f}m "
            f"{np.mean([r['median_err_tracking'] for r in valid]):>7.1f}m "
            f"{np.mean([r['p90_err_tracking'] for r in valid]):>7.1f}m "
            f"{np.mean([r['converge_s'] for r in valid if r['converge_s'] >= 0]):>6.1f}s "
            f"{np.mean([r['fine_rate_pct'] for r in valid]):>5.1f}%"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Multi-bag PF benchmark")
    parser.add_argument("--bags", default=str(PKG_DIR / "config" / "bags.yaml"),
                        help="Path to bags.yaml")
    parser.add_argument("--use-cache", action="store_true",
                        help="Replay PF from cached match files (no TRT)")
    parser.add_argument("--save-cache", action="store_true",
                        help="Save match cache after each full bag replay")
    parser.add_argument("--filter", nargs="+", metavar="BAG_NAME",
                        help="Run only these bag names (e.g. Day2.6 Day3.1)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit frames per bag (for debugging)")
    parser.add_argument("--altitude-min", type=float, default=None,
                        help="Override altitude_min_process_m for all bags")
    args = parser.parse_args()

    bags_yaml = yaml.safe_load(open(args.bags))
    base_cfg_path = PKG_DIR / bags_yaml["base_config"]
    base_cfg = load_config(str(base_cfg_path))

    rows = []
    for bag_entry in bags_yaml["bags"]:
        bag_name = bag_entry["name"]

        if args.filter and bag_name not in args.filter:
            print(f"[bench] Skipping {bag_name} (not in --filter)")
            continue

        merged_cfg = _merge_config(base_cfg, bag_entry)
        cache_file = PKG_DIR / "results" / bag_name / "match_cache.pkl"

        print(f"\n{'='*60}")
        print(f"[bench] Running {bag_name}")
        if bag_entry.get("notes"):
            print(f"  notes: {bag_entry['notes']}")
        print(f"{'='*60}")

        try:
            if args.use_cache:
                if not cache_file.exists():
                    print(f"[bench] No cache for {bag_name} — skipping. "
                          f"Run with --save-cache first.")
                    rows.append({
                        "bag": bag_name, "n_frames": 0,
                        "median_err_all": -1, "median_err_tracking": -1,
                        "p90_err_tracking": -1, "converge_s": -1, "fine_rate_pct": 0.0,
                    })
                    continue
                summary = run_replay(
                    merged_cfg,
                    cache_path=str(cache_file),
                    altitude_min_override=args.altitude_min,
                    bag_name=bag_name,
                )
            else:
                summary = run_replay(
                    merged_cfg,
                    save_cache=args.save_cache,
                    altitude_min_override=args.altitude_min,
                    max_frames=args.max_frames,
                    bag_name=bag_name,
                )
        except Exception as exc:
            print(f"[bench] ERROR on {bag_name}: {exc}")
            import traceback
            traceback.print_exc()
            rows.append({
                "bag": bag_name, "n_frames": 0,
                "median_err_all": -1, "median_err_tracking": -1,
                "p90_err_tracking": -1, "converge_s": -1, "fine_rate_pct": 0.0,
            })
            continue

        rows.append({
            "bag": bag_name,
            "n_frames": summary["n_frames"],
            "median_err_all": summary["median_err_all"],
            "median_err_tracking": summary["median_err_tracking"],
            "p90_err_tracking": summary.get("p90_err_tracking", -1),
            "converge_s": summary["converge_s"],
            "fine_rate_pct": summary["fine_rate_pct"],
        })

    # Summary table
    _print_table(rows)

    # Save CSV
    out_csv = PKG_DIR / "results" / "bench_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(str(out_csv), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[bench] Summary saved: {out_csv}")


if __name__ == "__main__":
    main()
