#!/usr/bin/env python3
"""bench_vlad_pca.py

Compare DINOv3 L10 VLAD with and without PCA dimensionality reduction.

Methods
-------
  vlad_6144   — DINOv3 L10 VLAD K=16, full 6144-dim descriptor
  vlad_pca512 — same, but PCA-whitened to 512-dim

PCA is fit on the patch DB descriptors, then applied to both DB and queries.

Usage
-----
  # Run (PCA is fit on the fly from the existing DB):
  python3 bench_vlad_pca.py

  # Quick test:
  python3 bench_vlad_pca.py --max-frames 50

  # Different PCA dim:
  python3 bench_vlad_pca.py --pca-dim 256
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── project paths ──────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_DIR     = SCRIPTS_DIR.parent
REPO_ROOT   = PKG_DIR.parent

sys.path.insert(0, str(PKG_DIR))
from particle_filter_loc.geo_utils import ENUFrame
from particle_filter_loc.vlad_extractor import DINOv3VLADExtractor

# ── fixed config ───────────────────────────────────────────────────────────────
DEFAULT_CFG = str(PKG_DIR / "config" / "pf_config.yaml")
GRAYSCALE   = True
DINOV3_IMAGE_SIZE = 256

BENCH_BAGS = [
    {"name": "Day2.6", "mcap": "/media/aniel/storage/bags_jetson/095c6d2c7237fe986c6240628b6ef8dd/Day2.6_0.mcap", "start_offset_s": 50.0},
    {"name": "Day3.3", "mcap": "/media/aniel/storage/bags_jetson/bea276eb8b0c91f69e638c5994519cba/Day3.3_0.mcap", "start_offset_s": 50.0},
    {"name": "Day4.2", "mcap": "/media/aniel/storage/bags_jetson/3b12c33fb6c0c824e5461dc5656404cc/Day4.2_0.mcap", "start_offset_s": 50.0},
]

METHOD_COLORS = {
    "vlad_6144":   "#2E7D32",  # dark green
    "vlad_pca512": "#E65100",  # deep orange
    "vlad_pca256": "#6A1B9A",  # purple
}


# ── PCA wrapper (GPU, fit on patch DB) ─────────────────────────────────────────

class PCAProjector:
    """PCA whitening on GPU using torch SVD."""

    def __init__(self, n_components: int, device: str = "cuda"):
        self.n_components = n_components
        self.device = torch.device(device)
        self.mean: Optional[torch.Tensor] = None
        self.components: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "PCAProjector":
        """Fit PCA on X [N, D]. Stores mean and top-k components."""
        X = X.float().to(self.device)
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean
        # SVD: X_centered = U @ S @ V^T, components = V[:, :k]
        _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        self.components = Vh[:self.n_components]  # [k, D]
        explained = (S[:self.n_components] ** 2).sum() / (S ** 2).sum()
        print(f"  PCA fit: {X.shape[1]}→{self.n_components} dims, "
              f"explained variance: {explained:.1%}")
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project X [N, D] → [N, k], L2-normalized."""
        X = X.float().to(self.device)
        projected = (X - self.mean) @ self.components.T
        return F.normalize(projected, dim=1)


# ── PCA-wrapped extractor ─────────────────────────────────────────────────────

class PCAVLADExtractor:
    """Wraps DINOv3VLADExtractor with PCA projection on output."""

    def __init__(self, base_extractor: DINOv3VLADExtractor, pca: PCAProjector):
        self.base = base_extractor
        self.pca = pca

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        return self.base.preprocess(frame_bgr)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        vlad = self.base(tensor)  # [1, 6144]
        return self.pca.transform(vlad)  # [1, pca_dim]


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_db(db_path: Path, names_path: Path) -> Tuple[torch.Tensor, List[str]]:
    data = torch.load(str(db_path), map_location="cuda", weights_only=False)
    if isinstance(data, dict):
        db = data.get("descriptors", data.get("boq_descriptors", list(data.values())[0]))
    else:
        db = data
    db = F.normalize(db.float().cuda(), dim=1)
    with open(str(names_path)) as f:
        names = [l.strip() for l in f if l.strip()]
    assert len(names) == db.shape[0]
    return db, names


def _patch_enu_array(names: List[str], gps_meta: dict, enu: ENUFrame) -> np.ndarray:
    coords = []
    for name in names:
        m = gps_meta.get(name)
        e, n = enu.wgs84_to_enu(m["lat"], m["lon"]) if m else (0.0, 0.0)
        coords.append((e, n))
    return np.array(coords, dtype=np.float64)


# ── Method container ───────────────────────────────────────────────────────────

class Method:
    def __init__(self, name: str, extractor, db: torch.Tensor,
                 patch_names: List[str], patch_enu: np.ndarray):
        self.name = name
        self.extractor = extractor
        self.db = db
        self.patch_names = patch_names
        self.patch_enu = patch_enu

    def retrieve(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, int, float]:
        tensor = self.extractor.preprocess(frame_bgr)
        with torch.no_grad():
            desc = F.normalize(self.extractor(tensor).float(), dim=1)
        sims = (desc @ self.db.T).squeeze(0).cpu().numpy()
        top1 = int(sims.argmax())
        return sims, top1, float(sims[top1])


# ── Per-frame metrics ──────────────────────────────────────────────────────────

def compute_metrics(method: Method, frame_bgr: np.ndarray,
                    rtk_e: float, rtk_n: float, gt_idx: int) -> dict:
    sims, top1_idx, top1_sim = method.retrieve(frame_bgr)
    top1_e, top1_n = method.patch_enu[top1_idx]
    error_m = math.hypot(top1_e - rtk_e, top1_n - rtk_n)
    top5 = set(np.argsort(sims)[::-1][:5].tolist())

    if gt_idx >= 0:
        cos_pos = float(sims[gt_idx])
        cos_neg = float(sims[top1_idx])
        rank_gt = int((sims > sims[gt_idx]).sum()) + 1
        r1 = int(top1_idx == gt_idx)
        r5 = int(gt_idx in top5)
    else:
        cos_pos = cos_neg = float("nan")
        rank_gt = r1 = r5 = -1

    return {
        "top1_patch": method.patch_names[top1_idx],
        "top1_e": top1_e, "top1_n": top1_n,
        "error_m": error_m,
        "cos_pos": cos_pos, "cos_neg": cos_neg,
        "rank_gt": rank_gt, "recall1": r1, "recall5": r5,
    }


def compute_summary(name: str, rows: List[dict]) -> dict:
    valid = [r for r in rows if r["recall1"] >= 0]
    errs = np.array([r["error_m"] for r in valid])
    cp   = np.array([r["cos_pos"] for r in valid], dtype=float)
    cn   = np.array([r["cos_neg"] for r in valid], dtype=float)
    r1   = np.array([r["recall1"] for r in valid])
    r5   = np.array([r["recall5"] for r in valid])
    return {
        "method":         name,
        "n_frames":       len(rows),
        "n_with_gt":      len(valid),
        "recall1_pct":    float(r1.mean() * 100) if len(r1) else float("nan"),
        "recall5_pct":    float(r5.mean() * 100) if len(r5) else float("nan"),
        "median_error_m": float(np.median(errs)) if len(errs) else float("nan"),
        "p90_error_m":    float(np.percentile(errs, 90)) if len(errs) else float("nan"),
        "mean_cos_pos":   float(np.nanmean(cp)) if len(cp) else float("nan"),
        "mean_cos_neg":   float(np.nanmean(cn)) if len(cn) else float("nan"),
        "mean_margin":    float(np.nanmean(cp - cn)) if len(cp) else float("nan"),
    }


def print_summary_table(summaries: List[dict], title: str = "") -> None:
    hdr = (f"{'method':<20} {'n_gt':>6} {'R@1%':>6} {'R@5%':>6} "
           f"{'med_err':>8} {'p90_err':>8} {'cos+':>6} {'cos-':>6} {'margin':>7}")
    sep = "─" * len(hdr)
    if title:
        print(f"\n{title}")
    print(f"\n{sep}\n{hdr}\n{sep}")
    for s in summaries:
        print(f"{s['method']:<20} {s['n_with_gt']:>6d} "
              f"{s['recall1_pct']:>5.1f}% {s['recall5_pct']:>5.1f}%  "
              f"{s['median_error_m']:>7.1f}m {s['p90_error_m']:>7.1f}m  "
              f"{s['mean_cos_pos']:>6.3f} {s['mean_cos_neg']:>6.3f} "
              f"{s['mean_margin']:>+7.3f}")
    print(sep)


# ── Trajectory plot ────────────────────────────────────────────────────────────

def save_plot(frame_records: List[dict], method_names: List[str],
              out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=13)
    ax_traj, ax_err, ax_cos = axes

    rtk_e   = np.array([r["rtk_e"]   for r in frame_records])
    rtk_n   = np.array([r["rtk_n"]   for r in frame_records])
    elapsed = np.array([r["elapsed_s"] for r in frame_records])

    ax_traj.scatter(rtk_e, rtk_n, c="black", s=4, label="RTK GT", zorder=5)
    for mn in method_names:
        c = METHOD_COLORS.get(mn, "gray")
        me_e = np.array([r[f"{mn}_top1_e"] for r in frame_records])
        me_n = np.array([r[f"{mn}_top1_n"] for r in frame_records])
        ax_traj.scatter(me_e, me_n, c=c, s=3, alpha=0.5, label=mn)
    ax_traj.set_xlabel("East (m)"); ax_traj.set_ylabel("North (m)")
    ax_traj.set_title("Trajectory (ENU)"); ax_traj.legend(fontsize=7, markerscale=3)
    ax_traj.set_aspect("equal"); ax_traj.grid(linewidth=0.4)

    for mn in method_names:
        c = METHOD_COLORS.get(mn, "gray")
        err = np.array([r[f"{mn}_error_m"] for r in frame_records])
        ax_err.plot(elapsed, err, color=c, linewidth=0.8, alpha=0.8, label=mn)
    ax_err.set_xlabel("Elapsed (s)"); ax_err.set_ylabel("Top-1 error (m)")
    ax_err.set_title("Error vs time"); ax_err.legend(fontsize=7)
    ax_err.grid(linewidth=0.4); ax_err.set_ylim(bottom=0)

    for mn in method_names:
        c = METHOD_COLORS.get(mn, "gray")
        cp = np.array([r[f"{mn}_cos_pos"] for r in frame_records], dtype=float)
        cn = np.array([r[f"{mn}_cos_neg"] for r in frame_records], dtype=float)
        ax_cos.plot(elapsed, cp, color=c, lw=0.9, label=f"{mn} +")
        ax_cos.plot(elapsed, cn, color=c, lw=0.9, ls="--", alpha=0.6, label=f"{mn} −")
    ax_cos.set_xlabel("Elapsed (s)"); ax_cos.set_ylabel("Cosine sim")
    ax_cos.set_title("cos_pos (solid) / cos_neg (dashed)")
    ax_cos.legend(fontsize=6); ax_cos.grid(linewidth=0.4); ax_cos.set_ylim(0, 1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ── Bag replay ─────────────────────────────────────────────────────────────────

def run_bag(bag_entry: dict, methods: List[Method], cfg: dict,
            gt_radius_m: float, max_frames: Optional[int],
            out_dir: Path) -> List[dict]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    bag_name    = bag_entry["name"]
    mcap_path   = bag_entry["mcap"]
    start_off   = bag_entry["start_offset_s"]
    rcfg        = cfg["replay"]
    cam_topic   = rcfg.get("camera_topic", "/camera/image_mono")
    rtk_topic   = rcfg.get("rtk_topic",    "/m300/rtk/fix")
    alt_topic   = rcfg.get("altimeter_topic", "/altimeter/range")
    alt_min     = rcfg.get("altitude_min_process_m", 0.0)

    enu_orig = cfg["enu_origin"]
    enu = ENUFrame(enu_orig["lat"], enu_orig["lon"])

    method_names = [m.name for m in methods]
    per_method_rows: Dict[str, List[dict]] = {m.name: [] for m in methods}
    frame_records: List[dict] = []

    bag_out = out_dir / bag_name
    bag_out.mkdir(parents=True, exist_ok=True)
    base_fields = ["frame_idx", "elapsed_s", "rtk_lat", "rtk_lon",
                   "rtk_e", "rtk_n", "gt_patch", "has_gt", "altitude_m"]
    mfields = []
    for mn in method_names:
        mfields += [f"{mn}_top1", f"{mn}_error_m", f"{mn}_cos_pos",
                    f"{mn}_cos_neg", f"{mn}_rank_gt", f"{mn}_r1", f"{mn}_r5"]
    csv_path = bag_out / "per_frame.csv"

    print(f"\n[{bag_name}] Opening: {mcap_path}")

    gt_lat = gt_lon = None
    altitude_buf = deque(maxlen=10)
    current_alt  = 0.0
    t_start      = None
    frame_idx    = 0

    with open(str(csv_path), "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=base_fields + mfields)
        writer.writeheader()

        with open(str(mcap_path), "rb") as bag_f:
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
                if max_frames and frame_idx > max_frames:
                    print(f"  [{bag_name}] max_frames={max_frames} reached, stopping.")
                    break
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

                ref = methods[0]
                dx = ref.patch_enu[:, 0] - rtk_e
                dy = ref.patch_enu[:, 1] - rtk_n
                dists = np.hypot(dx, dy)
                gt_idx_ref = int(dists.argmin())
                has_gt = dists[gt_idx_ref] <= gt_radius_m
                gt_name = ref.patch_names[gt_idx_ref] if has_gt else ""

                row: dict = {
                    "frame_idx": frame_idx,
                    "elapsed_s": round(elapsed_s, 3),
                    "rtk_lat": round(gt_lat, 8), "rtk_lon": round(gt_lon, 8),
                    "rtk_e": round(rtk_e, 2), "rtk_n": round(rtk_n, 2),
                    "gt_patch": gt_name, "has_gt": int(has_gt),
                    "altitude_m": round(current_alt, 1),
                }
                traj_row = {"elapsed_s": elapsed_s, "rtk_e": rtk_e, "rtk_n": rtk_n}

                t0 = time.monotonic()
                for method in methods:
                    gt_idx = -1
                    if has_gt:
                        try:
                            gt_idx = method.patch_names.index(gt_name)
                        except ValueError:
                            pass
                    m = compute_metrics(method, frame_bgr, rtk_e, rtk_n, gt_idx)
                    mn = method.name
                    row[f"{mn}_top1"]    = m["top1_patch"]
                    row[f"{mn}_error_m"] = round(m["error_m"], 2)
                    row[f"{mn}_cos_pos"] = round(m["cos_pos"], 4) if not math.isnan(m["cos_pos"]) else ""
                    row[f"{mn}_cos_neg"] = round(m["cos_neg"], 4)
                    row[f"{mn}_rank_gt"] = m["rank_gt"]
                    row[f"{mn}_r1"]      = m["recall1"]
                    row[f"{mn}_r5"]      = m["recall5"]
                    per_method_rows[mn].append(m)
                    traj_row[f"{mn}_top1_e"]  = m["top1_e"]
                    traj_row[f"{mn}_top1_n"]  = m["top1_n"]
                    traj_row[f"{mn}_error_m"] = m["error_m"]
                    traj_row[f"{mn}_cos_pos"] = m["cos_pos"]
                    traj_row[f"{mn}_cos_neg"] = m["cos_neg"]

                dt_ms = (time.monotonic() - t0) * 1000
                frame_records.append(traj_row)
                writer.writerow(row)

                if frame_idx % 100 == 1:
                    err_str = "  ".join(
                        f"{m.name}:{per_method_rows[m.name][-1]['error_m']:.0f}m"
                        for m in methods
                    )
                    print(f"  frame {frame_idx:4d}  t={elapsed_s:.1f}s "
                          f"alt={current_alt:.0f}m  gt={'Y' if has_gt else 'N'}  "
                          f"{err_str}  [{dt_ms:.0f}ms]")

    print(f"  [{bag_name}] {frame_idx} frames  → {csv_path}")

    summaries = [compute_summary(mn, per_method_rows[mn]) for mn in method_names]
    print_summary_table(summaries, title=f"  ── {bag_name}")

    if frame_records:
        save_plot(frame_records, method_names,
                  bag_out / "trajectory.png",
                  f"VLAD PCA benchmark — {bag_name}")

    sum_path = bag_out / "summary.csv"
    with open(str(sum_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)

    return summaries


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DINOv3 L10 VLAD with/without PCA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CFG)
    parser.add_argument("--pca-dim", dest="pca_dim", type=int, default=512)
    parser.add_argument("--gt-radius", dest="gt_radius", type=float, default=50.0)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    args = parser.parse_args()

    cfg   = _load_config(args.config)
    mcfg  = cfg["matchers"]
    script_dir  = Path(mcfg["script_dir"])
    patches_dir = script_dir / mcfg.get("patches_dir", "data/patches")
    meta_path   = script_dir / mcfg.get("gps_metadata_path", "data/patches/gps_metadata.json")

    with open(str(meta_path)) as f:
        gps_meta = json.load(f)
    enu = ENUFrame(cfg["enu_origin"]["lat"], cfg["enu_origin"]["lon"])

    pca_dim = args.pca_dim
    out_dir = PKG_DIR / "results" / f"bench_vlad_pca{pca_dim}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load base VLAD DB ─────────────────────────────────────────────────
    dinov3_db_dir = script_dir / "data" / "descriptors_vlad_l10_aerial"
    db_pt   = dinov3_db_dir / "vlad_descriptors.pt"
    cb_pt   = dinov3_db_dir / "vlad_codebook.pt"
    n_path  = dinov3_db_dir / "patch_names.txt"

    if not db_pt.exists() or not cb_pt.exists():
        print(f"[error] DINOv3 L10 DB not found at {dinov3_db_dir}")
        return

    db_full, names = _load_db(db_pt, n_path)
    patch_enu = _patch_enu_array(names, gps_meta, enu)
    print(f"Loaded DB: {db_full.shape}  ({names[0]} … {names[-1]})")

    # ── fit PCA on patch DB ───────────────────────────────────────────────
    print(f"\nFitting PCA: {db_full.shape[1]} → {pca_dim} dims …")
    pca = PCAProjector(n_components=pca_dim, device="cuda")
    pca.fit(db_full)
    db_pca = pca.transform(db_full)
    print(f"PCA DB: {db_pca.shape}")

    # ── load extractor (shared backbone, two output paths) ────────────────
    print("\nLoading DINOv3 L10 extractor …")
    ext_base = DINOv3VLADExtractor(
        str(cb_pt), layer_idx=10,
        image_size=DINOV3_IMAGE_SIZE, grayscale=GRAYSCALE, device="cuda",
    )
    ext_pca = PCAVLADExtractor(ext_base, pca)

    # ── build methods ─────────────────────────────────────────────────────
    methods: List[Method] = [
        Method("vlad_6144", ext_base, db_full, names, patch_enu),
        Method(f"vlad_pca{pca_dim}", ext_pca, db_pca, names, patch_enu),
    ]

    print(f"\nMethods: {[m.name for m in methods]}")
    print(f"  vlad_6144:   DB {db_full.shape}, mem {db_full.nbytes/1e6:.1f} MB")
    print(f"  vlad_pca{pca_dim}: DB {db_pca.shape}, mem {db_pca.nbytes/1e6:.1f} MB")

    # ── run bags ──────────────────────────────────────────────────────────
    all_summaries: Dict[str, List[dict]] = {}

    for bag_entry in BENCH_BAGS:
        sums = run_bag(bag_entry, methods, cfg, args.gt_radius, args.max_frames, out_dir)
        all_summaries[bag_entry["name"]] = sums

    # ── combined table ────────────────────────────────────────────────────
    method_names = [m.name for m in methods]
    print(f"\n\n{'='*70}")
    print("COMBINED — all bags")
    print(f"{'='*70}")
    hdr = (f"{'bag':<10} {'method':<20} {'R@1%':>6} {'R@5%':>6} "
           f"{'med_err':>8} {'p90_err':>8} {'cos+':>6} {'cos-':>6} {'margin':>7}")
    sep = "─" * len(hdr)
    print(hdr); print(sep)

    combined_rows = []
    for bag_name, sums in all_summaries.items():
        for s in sums:
            print(f"{bag_name:<10} {s['method']:<20} "
                  f"{s['recall1_pct']:>5.1f}% {s['recall5_pct']:>5.1f}%  "
                  f"{s['median_error_m']:>7.1f}m {s['p90_error_m']:>7.1f}m  "
                  f"{s['mean_cos_pos']:>6.3f} {s['mean_cos_neg']:>6.3f} "
                  f"{s['mean_margin']:>+7.3f}")
            combined_rows.append({"bag": bag_name, **s})
        print(sep)

    print(f"\nMEAN across {len(BENCH_BAGS)} bags:")
    print(hdr); print(sep)
    for mn in method_names:
        mn_rows = [s for sums in all_summaries.values() for s in sums if s["method"] == mn]
        r1  = np.nanmean([r["recall1_pct"]    for r in mn_rows])
        r5  = np.nanmean([r["recall5_pct"]    for r in mn_rows])
        med = np.nanmean([r["median_error_m"] for r in mn_rows])
        p90 = np.nanmean([r["p90_error_m"]    for r in mn_rows])
        cp  = np.nanmean([r["mean_cos_pos"]   for r in mn_rows])
        cn  = np.nanmean([r["mean_cos_neg"]   for r in mn_rows])
        mg  = np.nanmean([r["mean_margin"]    for r in mn_rows])
        print(f"{'MEAN':<10} {mn:<20} "
              f"{r1:>5.1f}% {r5:>5.1f}%  "
              f"{med:>7.1f}m {p90:>7.1f}m  "
              f"{cp:>6.3f} {cn:>6.3f} {mg:>+7.3f}")
    print(sep)

    if combined_rows:
        f_out = out_dir / "combined_summary.csv"
        with open(str(f_out), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
            w.writeheader(); w.writerows(combined_rows)
        print(f"\n[done] Combined CSV: {f_out}")


if __name__ == "__main__":
    main()
