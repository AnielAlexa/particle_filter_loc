#!/usr/bin/env python3
"""PCA512 + temperature scaling benchmark on all 3 bags.

Compares VLAD 6144 vs PCA512 with different temperature values.
Reports retrieval metrics + softmax peakiness (relevant for particle filter).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_DIR     = SCRIPTS_DIR.parent
sys.path.insert(0, str(PKG_DIR))

from particle_filter_loc.geo_utils import ENUFrame
from particle_filter_loc.vlad_extractor import DINOv3VLADExtractor

BENCH_BAGS = [
    {"name": "Day2.1", "mcap": "/media/aniel/storage/bags_jetson/c6f4f561239128c653ca28007877f6a1/Day2.1_0.mcap", "start_offset_s": 50.0},
    {"name": "Day2.2", "mcap": "/media/aniel/storage/bags_jetson/eaf2552402e96b1c2d6a7b92a666a10c/Day2.2_0.mcap", "start_offset_s": 50.0},
    {"name": "Day2.4", "mcap": "/media/aniel/storage/bags_jetson/05e5e3545cbe8ff300a5b5bbb2eff4be/Day2.4_80_0.mcap", "start_offset_s": 50.0},
    {"name": "Day2.5", "mcap": "/media/aniel/storage/bags_jetson/f4aec31f0ee5390ac105ff7d69712e7c/Day2.5_0.mcap", "start_offset_s": 50.0},
    {"name": "Day2.5_70", "mcap": "/media/aniel/storage/bags_jetson/31b7e1ed1ccb139c120a07fcd8fa4def/Day2.5_70s_0.mcap", "start_offset_s": 50.0},
    {"name": "Day2.6", "mcap": "/media/aniel/storage/bags_jetson/095c6d2c7237fe986c6240628b6ef8dd/Day2.6_0.mcap", "start_offset_s": 50.0},
    {"name": "Day3.1", "mcap": "/media/aniel/storage/bags_jetson/1f24cb6af4e6fea2d9d31829715aad95/Day3.1_0.mcap", "start_offset_s": 50.0},
    {"name": "Day3.2", "mcap": "/media/aniel/storage/bags_jetson/7e84212b99537e93c8dab7cdde6ca76b/Day3.2_0.mcap", "start_offset_s": 50.0},
    {"name": "Day3.3", "mcap": "/media/aniel/storage/bags_jetson/bea276eb8b0c91f69e638c5994519cba/Day3.3_0.mcap", "start_offset_s": 50.0},
    {"name": "Day4.2", "mcap": "/media/aniel/storage/bags_jetson/3b12c33fb6c0c824e5461dc5656404cc/Day4.2_0.mcap", "start_offset_s": 50.0},
    {"name": "Day4.3", "mcap": "/media/aniel/storage/bags_jetson/991dce18c6051d7043042d2482c94455/Day4.3_0.mcap", "start_offset_s": 50.0},
]

TEMPERATURES = [1.0, 0.2, 0.1, 0.05]


def run_bag(bag_entry: dict, cfg: dict, ext, db_full, db_pca, pca_mean,
            pca_components, patch_names, patch_enu, gt_radius_m: float):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    bag_name  = bag_entry["name"]
    mcap_path = bag_entry["mcap"]
    start_off = bag_entry["start_offset_s"]
    rcfg      = cfg["replay"]
    cam_topic = rcfg.get("camera_topic", "/camera/image_mono")
    rtk_topic = rcfg.get("rtk_topic", "/m300/rtk/fix")
    alt_topic = rcfg.get("altimeter_topic", "/altimeter/range")
    alt_min   = rcfg.get("altitude_min_process_m", 0.0)

    enu_orig = cfg["enu_origin"]
    enu = ENUFrame(enu_orig["lat"], enu_orig["lon"])

    # Per-frame collectors
    rows_vlad: List[dict] = []
    rows_pca: List[dict] = []
    peak_stats = {method: {t: [] for t in TEMPERATURES} for method in ["vlad", "pca"]}

    gt_lat = gt_lon = None
    altitude_buf = deque(maxlen=10)
    current_alt = 0.0
    t_start = None
    frame_idx = 0

    print(f"\n[{bag_name}] Opening: {mcap_path}")

    with open(mcap_path, "rb") as bag_f:
        reader = make_reader(bag_f, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded_msg in reader.iter_decoded_messages(
                topics=[cam_topic, rtk_topic, alt_topic]):
            topic = channel.topic
            ts_ns = message.log_time
            if t_start is None:
                t_start = ts_ns
            elapsed_s = (ts_ns - t_start) * 1e-9
            if elapsed_s < start_off:
                continue

            if topic == alt_topic:
                altitude_buf.append(float(decoded_msg.range))
                current_alt = float(np.median(altitude_buf))
                continue
            if topic == rtk_topic:
                gt_lat = decoded_msg.latitude
                gt_lon = decoded_msg.longitude
                continue
            if topic != cam_topic:
                continue

            frame_idx += 1
            if alt_min > 0 and current_alt < alt_min:
                continue
            if gt_lat is None:
                continue

            h, w = decoded_msg.height, decoded_msg.width
            if decoded_msg.encoding == "mono8":
                gray = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(h, w)
                frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif decoded_msg.encoding in ("bgr8", "rgb8"):
                frame_bgr = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(h, w, 3)
                if decoded_msg.encoding == "rgb8":
                    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            else:
                continue

            rtk_e, rtk_n = enu.wgs84_to_enu(gt_lat, gt_lon)

            # GT patch
            dx = patch_enu[:, 0] - rtk_e
            dy = patch_enu[:, 1] - rtk_n
            dists = np.hypot(dx, dy)
            gt_idx = int(dists.argmin())
            has_gt = dists[gt_idx] <= gt_radius_m
            if not has_gt:
                continue

            tensor = ext.preprocess(frame_bgr)
            with torch.no_grad():
                vlad_desc = F.normalize(ext(tensor).float(), dim=1)

            # VLAD 6144
            sims_vlad = (vlad_desc @ db_full.T).squeeze(0)
            top1_vlad = int(sims_vlad.argmax())
            e_vlad = math.hypot(patch_enu[top1_vlad, 0] - rtk_e, patch_enu[top1_vlad, 1] - rtk_n)
            rows_vlad.append({
                "error_m": e_vlad,
                "recall1": int(top1_vlad == gt_idx),
                "recall5": int(gt_idx in set(sims_vlad.topk(5).indices.tolist())),
                "cos_pos": sims_vlad[gt_idx].item(),
                "cos_neg": sims_vlad[top1_vlad].item(),
            })

            # PCA512
            desc_pca = F.normalize((vlad_desc - pca_mean) @ pca_components.T, dim=1)
            sims_pca = (desc_pca @ db_pca.T).squeeze(0)
            top1_pca = int(sims_pca.argmax())
            e_pca = math.hypot(patch_enu[top1_pca, 0] - rtk_e, patch_enu[top1_pca, 1] - rtk_n)
            rows_pca.append({
                "error_m": e_pca,
                "recall1": int(top1_pca == gt_idx),
                "recall5": int(gt_idx in set(sims_pca.topk(5).indices.tolist())),
                "cos_pos": sims_pca[gt_idx].item(),
                "cos_neg": sims_pca[top1_pca].item(),
            })

            # Peakiness at different temperatures
            for tau in TEMPERATURES:
                w_vlad = F.softmax(sims_vlad / tau, dim=0)
                w_pca  = F.softmax(sims_pca / tau, dim=0)
                # Weight on GT patch
                peak_stats["vlad"][tau].append({
                    "peak": w_vlad.max().item(),
                    "gt_weight": w_vlad[gt_idx].item(),
                    "top5_mass": w_vlad.topk(5).values.sum().item(),
                    "entropy": -(w_vlad * w_vlad.log()).sum().item(),
                })
                peak_stats["pca"][tau].append({
                    "peak": w_pca.max().item(),
                    "gt_weight": w_pca[gt_idx].item(),
                    "top5_mass": w_pca.topk(5).values.sum().item(),
                    "entropy": -(w_pca * w_pca.log()).sum().item(),
                })

            if frame_idx % 500 == 1:
                print(f"  frame {frame_idx:5d}  t={elapsed_s:.1f}s  "
                      f"vlad:{e_vlad:.0f}m  pca:{e_pca:.0f}m")

    n = len(rows_vlad)
    print(f"  [{bag_name}] {n} frames with GT")

    return bag_name, n, rows_vlad, rows_pca, peak_stats


def summarize(name: str, rows: List[dict]) -> dict:
    if not rows:
        return {"method": name, "n": 0, "R@1": float("nan"), "R@5": float("nan"),
                "med_err": float("nan"), "p90_err": float("nan"),
                "cos+": float("nan"), "cos-": float("nan"), "margin": float("nan")}
    errs = np.array([r["error_m"] for r in rows])
    r1 = np.mean([r["recall1"] for r in rows]) * 100
    r5 = np.mean([r["recall5"] for r in rows]) * 100
    cp = np.mean([r["cos_pos"] for r in rows])
    cn = np.mean([r["cos_neg"] for r in rows])
    return {
        "method": name,
        "n": len(rows),
        "R@1": r1, "R@5": r5,
        "med_err": float(np.median(errs)),
        "p90_err": float(np.percentile(errs, 90)),
        "cos+": cp, "cos-": cn,
        "margin": cp - cn,
    }


def main():
    cfg_path = str(PKG_DIR / "config" / "pf_config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["matchers"]
    script_dir = Path(mcfg["script_dir"])
    db_dir = script_dir / "data" / "descriptors_vlad_l10_aerial"
    meta_path = script_dir / mcfg.get("gps_metadata_path", "data/patches/gps_metadata.json")

    with open(str(meta_path)) as f:
        gps_meta = json.load(f)
    enu = ENUFrame(cfg["enu_origin"]["lat"], cfg["enu_origin"]["lon"])

    # Load DB + names
    db_full = torch.load(str(db_dir / "vlad_descriptors.pt"), map_location="cuda", weights_only=False)
    db_full = F.normalize(db_full.float().cuda(), dim=1)
    with open(str(db_dir / "patch_names.txt")) as f:
        patch_names = [l.strip() for l in f if l.strip()]

    patch_enu = np.array([
        enu.wgs84_to_enu(gps_meta[n]["lat"], gps_meta[n]["lon"]) if n in gps_meta else (0, 0)
        for n in patch_names
    ], dtype=np.float64)

    print(f"DB: {db_full.shape}, patches: {len(patch_names)}")

    # Fit PCA512
    pca_mean = db_full.mean(dim=0)
    X_c = db_full - pca_mean
    _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
    pca_dim = 512
    pca_components = Vh[:pca_dim]
    explained = (S[:pca_dim] ** 2).sum() / (S ** 2).sum()
    print(f"PCA: 6144→{pca_dim}, explained: {explained:.1%}")
    db_pca = F.normalize(X_c @ pca_components.T, dim=1)

    # Load extractor
    ext = DINOv3VLADExtractor(
        str(db_dir / "vlad_codebook.pt"), layer_idx=10,
        image_size=256, grayscale=True, device="cuda",
    )

    # Run all bags
    all_vlad, all_pca = [], []
    all_peaks = {"vlad": {t: [] for t in TEMPERATURES}, "pca": {t: [] for t in TEMPERATURES}}
    bag_results = []

    for bag_entry in BENCH_BAGS:
        try:
            bag_name, n, rows_v, rows_p, peaks = run_bag(
                bag_entry, cfg, ext, db_full, db_pca, pca_mean,
                pca_components, patch_names, patch_enu, gt_radius_m=50.0,
            )
        except Exception as e:
            print(f"  [{bag_entry['name']}] ERROR: {e} — skipping")
            continue
        if n == 0:
            print(f"  [{bag_name}] skipped — no GT frames")
            continue
        all_vlad.extend(rows_v)
        all_pca.extend(rows_p)
        for method in ["vlad", "pca"]:
            for tau in TEMPERATURES:
                all_peaks[method][tau].extend(peaks[method][tau])

        sv = summarize("vlad_6144", rows_v)
        sp = summarize("pca512", rows_p)
        bag_results.append((bag_name, sv, sp, peaks))

    # ── Per-bag retrieval results ─────────────────────────────────────────
    print(f"\n{'='*85}")
    print("RETRIEVAL METRICS — per bag")
    print(f"{'='*85}")
    hdr = f"{'bag':<10} {'method':<12} {'R@1':>6} {'R@5':>6} {'med_err':>8} {'p90_err':>8} {'cos+':>6} {'cos-':>6} {'margin':>7}"
    print(hdr)
    print("─" * 85)
    for bag_name, sv, sp, _ in bag_results:
        for s in [sv, sp]:
            print(f"{bag_name:<10} {s['method']:<12} {s['R@1']:>5.1f}% {s['R@5']:>5.1f}% "
                  f"{s['med_err']:>7.1f}m {s['p90_err']:>7.1f}m "
                  f"{s['cos+']:>6.3f} {s['cos-']:>6.3f} {s['margin']:>+7.3f}")
        print("─" * 85)

    # Mean
    sv_all = summarize("vlad_6144", all_vlad)
    sp_all = summarize("pca512", all_pca)
    print("COMBINED:")
    for s in [sv_all, sp_all]:
        print(f"{'ALL':<10} {s['method']:<12} {s['R@1']:>5.1f}% {s['R@5']:>5.1f}% "
              f"{s['med_err']:>7.1f}m {s['p90_err']:>7.1f}m "
              f"{s['cos+']:>6.3f} {s['cos-']:>6.3f} {s['margin']:>+7.3f}")
    print("─" * 85)

    # ── Peakiness at different temperatures ───────────────────────────────
    print(f"\n{'='*85}")
    print("SOFTMAX PEAKINESS — all bags combined")
    print(f"{'='*85}")
    print(f"{'tau':<8} {'method':<12} {'peak':>8} {'gt_weight':>10} {'top5_mass':>10} {'entropy':>10}")
    print("─" * 62)
    for tau in TEMPERATURES:
        for mname, mkey in [("vlad_6144", "vlad"), ("pca512", "pca")]:
            rows = all_peaks[mkey][tau]
            peak = np.mean([r["peak"] for r in rows])
            gt_w = np.mean([r["gt_weight"] for r in rows])
            top5 = np.mean([r["top5_mass"] for r in rows])
            ent  = np.mean([r["entropy"] for r in rows])
            print(f"{tau:<8.2f} {mname:<12} {peak:>8.4f} {gt_w:>10.4f} {top5:>10.4f} {ent:>10.2f}")
        print("─" * 62)

    print(f"\nUniform baseline: peak={1/700:.4f}, gt_weight={1/700:.4f}, "
          f"top5={5/700:.4f}, entropy={np.log(700):.2f}")
    print(f"\ngt_weight = softmax weight assigned to the GT patch (higher = PF gets better signal)")


if __name__ == "__main__":
    main()
