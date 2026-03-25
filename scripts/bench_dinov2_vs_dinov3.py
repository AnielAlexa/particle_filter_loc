#!/usr/bin/env python3
"""bench_dinov2_vs_dinov3.py

Compare DINOv2 (ViT-S/14, torch.hub) vs DINOv3 (ViT-S/16, timm) using the
same VLAD pipeline with aerial codebooks.

Methods
-------
  dinov2_l10 — DINOv2 ViT-S/14, 224×224, layer 10, aerial codebook
  dinov2_l11 — DINOv2 ViT-S/14, 224×224, layer 11, aerial codebook
  dinov3_l10 — DINOv3 ViT-S/16, 256×256, layer 10, aerial codebook
  dinov3_l11 — DINOv3 ViT-S/16, 256×256, layer 11, aerial codebook

All: grayscale, VLAD K=16, value facet, aerial codebook (10k images from JSON).

Usage
-----
  # Build DINOv2 codebooks + DBs, then bench all 4:
  python3 bench_dinov2_vs_dinov3.py --build-codebooks --max-json-images 10000

  # Quick test:
  python3 bench_dinov2_vs_dinov3.py --build-codebooks --max-json-images 2000 --max-frames 50

  # Run only (codebooks already built):
  python3 bench_dinov2_vs_dinov3.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
DATA_ROOT   = Path("/media/aniel/storage/dataset_to_analize")
AERIAL_JSON = DATA_ROOT / "data" / "aerialvl100m_multi_vpair_train.json"

DEFAULT_CFG = str(PKG_DIR / "config" / "pf_config.yaml")
K_CLUSTERS  = 16
GRAYSCALE   = True

# DINOv2: 224×224 input (14×16 = 224), 256 patch tokens, 1 CLS prefix
DINOV2_IMAGE_SIZE = 224
DINOV2_NUM_PREFIX = 1

# DINOv3: 256×256 input (16×16 = 256), 256 patch tokens
DINOV3_IMAGE_SIZE = 256

BENCH_BAGS = [
    {"name": "Day2.6", "mcap": "/media/aniel/storage/bags_jetson/095c6d2c7237fe986c6240628b6ef8dd/Day2.6_0.mcap", "start_offset_s": 50.0},
    {"name": "Day3.3", "mcap": "/media/aniel/storage/bags_jetson/bea276eb8b0c91f69e638c5994519cba/Day3.3_0.mcap", "start_offset_s": 50.0},
    {"name": "Day4.2", "mcap": "/media/aniel/storage/bags_jetson/3b12c33fb6c0c824e5461dc5656404cc/Day4.2_0.mcap", "start_offset_s": 50.0},
]

METHOD_COLORS = {
    "dinov2_l10": "#1565C0",  # dark blue
    "dinov2_l11": "#42A5F5",  # light blue
    "dinov3_l10": "#2E7D32",  # dark green
    "dinov3_l11": "#81C784",  # light green
}

# ImageNet mean/std for DINOv2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── DINOv2 VLAD extractor ─────────────────────────────────────────────────────

class DINOv2VLADExtractor:
    """DINOv2 ViT-S/14 + VLAD extractor. Mirrors DINOv3VLADExtractor interface."""

    def __init__(
        self,
        codebook_path: Optional[str] = None,
        layer_idx: int = 10,
        image_size: int = 224,
        grayscale: bool = True,
        device: str = "cuda",
    ) -> None:
        self.layer_idx = layer_idx
        self.image_size = image_size
        self.grayscale = grayscale
        self.device = torch.device(device)
        self._hook_out: Optional[torch.Tensor] = None
        self.num_prefix_tokens = DINOV2_NUM_PREFIX  # CLS only, no registers

        # ── backbone ────────────────────────────────────────────────────
        print(f"Loading dinov2_vits14 (torch.hub, img_size={image_size}) …")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval().to(self.device)

        # Normalization: ImageNet mean/std
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32,
                                 device=self.device).view(1, 3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD, dtype=torch.float32,
                                 device=self.device).view(1, 3, 1, 1)

        # ── QKV hook ────────────────────────────────────────────────────
        self._hook_handle = (
            self.model.blocks[layer_idx].attn.qkv
            .register_forward_hook(self._qkv_hook)
        )
        print(f"Registered QKV hook on block {layer_idx} (value facet), "
              f"num_prefix_tokens={self.num_prefix_tokens}")

        # ── codebook ────────────────────────────────────────────────────
        self.c_centers: Optional[torch.Tensor] = None
        self.K: int = 0
        self.D: int = 0

        if codebook_path is not None:
            self._load_codebook(codebook_path)

    def _qkv_hook(self, module, inputs, output: torch.Tensor) -> None:
        B, N, _ = output.shape
        D = output.shape[-1] // 3
        self._hook_out = output.reshape(B, N, 3, D)[:, :, 2, :]

    def _load_codebook(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        if isinstance(ck, dict):
            self.c_centers = ck["c_centers"].float().to(self.device)
        else:
            self.c_centers = ck.float().to(self.device)
        self.K = self.c_centers.shape[0]
        self.D = self.c_centers.shape[1]
        print(f"VLAD codebook loaded: K={self.K}, D={self.D}  "
              f"(descriptor dim = {self.K * self.D})")

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        frame = frame_bgr[y0:y0 + s, x0:x0 + s]
        frame = cv2.resize(frame, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_CUBIC)
        if self.grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.stack([gray, gray, gray], axis=2)
        else:
            frame = frame[:, :, ::-1]
        tensor = (
            torch.from_numpy(frame.transpose(2, 0, 1).copy())
            .unsqueeze(0).float().to(self.device) / 255.0
        )
        return (tensor - self.mean) / self.std

    def extract_patch_features(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self._hook_out = None
            self.model(tensor)
            assert self._hook_out is not None, "QKV hook did not fire"
            feats = self._hook_out[:, self.num_prefix_tokens:, :]
            feats = F.normalize(feats, dim=-1)
        return feats

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        feats = self.extract_patch_features(tensor)
        return self._vlad_aggregate(feats)

    def _vlad_aggregate(self, feats: torch.Tensor) -> torch.Tensor:
        assert self.c_centers is not None
        B, N, D = feats.shape
        K = self.K
        c = self.c_centers
        c_sq = (c ** 2).sum(-1)
        cross = torch.einsum("bnd,kd->bnk", feats, c)
        dists = 1.0 + c_sq - 2.0 * cross
        assignments = dists.argmin(dim=-1)
        vlad = feats.new_zeros(B, K, D)
        for k in range(K):
            mask = (assignments == k).unsqueeze(-1).float()
            residuals = feats - c[k].unsqueeze(0).unsqueeze(0)
            cluster_sum = (residuals * mask).sum(dim=1)
            vlad[:, k] = F.normalize(cluster_sum, dim=-1)
        vlad = vlad.flatten(1)
        return F.normalize(vlad, dim=-1)

    def __del__(self) -> None:
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()


# ── helpers (from bench_vlad_codebook.py) ──────────────────────────────────────

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


def _collect_image_paths(json_path: Path, max_images: int, seed: int = 42) -> List[Path]:
    with open(str(json_path)) as f:
        entries = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(entries)
    paths: List[Path] = []
    for e in entries:
        if len(paths) >= max_images:
            break
        drone_path = DATA_ROOT / e["drone_img_dir"] / e["drone_img_name"]
        if drone_path.exists():
            paths.append(drone_path)
        if len(paths) >= max_images:
            break
        for sname in e.get("pair_pos_semipos_sate_img_list", []):
            sate_path = DATA_ROOT / e["sate_img_dir"] / sname
            if sate_path.exists():
                paths.append(sate_path)
            if len(paths) >= max_images:
                break
    print(f"  Collected {len(paths)} image paths from JSON (cap={max_images})")
    return paths


# ── codebook builder (generic — works with any extractor) ─────────────────────

def build_aerial_codebook(
    out_dir: Path,
    layer_idx: int,
    image_paths: List[Path],
    extractor,
    batch_size: int = 8,
    max_kmeans_samples: int = 200_000,
    k_clusters: int = K_CLUSTERS,
) -> None:
    try:
        import fast_pytorch_kmeans as fpk
    except ImportError:
        raise ImportError("pip install fast-pytorch-kmeans")
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    codebook_path = out_dir / "vlad_codebook.pt"

    print(f"\n[build_codebook l{layer_idx}] Extracting features from {len(image_paths)} images …")

    all_feats: list[torch.Tensor] = []
    failed = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"features l{layer_idx}"):
        batch = image_paths[i : i + batch_size]
        tensors = []
        for p in batch:
            img = cv2.imread(str(p))
            if img is None:
                failed += 1
                continue
            tensors.append(extractor.preprocess(img))
        if not tensors:
            continue
        batch_t = torch.cat(tensors, dim=0)
        feats = extractor.extract_patch_features(batch_t)
        all_feats.append(feats.cpu())

    if failed:
        print(f"  [warn] {failed} images could not be read")

    all_feats_t = torch.cat(all_feats, dim=0)
    N, T, D = all_feats_t.shape
    print(f"  Features: {all_feats_t.shape}  (total tokens = {N*T})")

    flat = all_feats_t.reshape(-1, D)
    if flat.shape[0] > max_kmeans_samples:
        idx = torch.randperm(flat.shape[0])[:max_kmeans_samples]
        flat = flat[idx]
    flat = flat.cuda()

    print(f"  K-means: {flat.shape[0]} vectors, K={k_clusters}, D={D}")
    kmeans = fpk.KMeans(n_clusters=k_clusters, mode="euclidean", verbose=1)
    kmeans.fit(flat)
    c_centers = kmeans.centroids.cpu().float()
    torch.save({"c_centers": c_centers}, str(codebook_path))
    print(f"  Codebook saved: {codebook_path}  shape={c_centers.shape}")


def build_patch_db(
    patches_dir: Path,
    out_dir: Path,
    extractor,
    batch_size: int = 8,
) -> None:
    from tqdm import tqdm

    codebook_path = out_dir / "vlad_codebook.pt"
    db_path       = out_dir / "vlad_descriptors.pt"
    names_path    = out_dir / "patch_names.txt"

    if db_path.exists():
        print(f"  [skip] DB already exists: {db_path}")
        return

    # Load codebook into extractor if not already loaded
    if extractor.c_centers is None:
        extractor._load_codebook(str(codebook_path))

    patch_paths = sorted(patches_dir.glob("*.png"))
    if not patch_paths:
        raise FileNotFoundError(f"No patches in {patches_dir}")

    print(f"  Building patch DB: {len(patch_paths)} patches")

    all_vlad: list[torch.Tensor] = []
    patch_names: list[str] = []

    for i in tqdm(range(0, len(patch_paths), batch_size), desc="patch DB"):
        batch = patch_paths[i : i + batch_size]
        tensors, names = [], []
        for p in batch:
            img = cv2.imread(str(p))
            if img is None:
                continue
            tensors.append(extractor.preprocess(img))
            names.append(p.stem)
        if not tensors:
            continue
        batch_t = torch.cat(tensors, dim=0)
        feats = extractor.extract_patch_features(batch_t)
        for j in range(feats.shape[0]):
            v = extractor._vlad_aggregate(feats[j].unsqueeze(0).to(extractor.device))
            all_vlad.append(v.cpu())
        patch_names.extend(names)

    vlad_db = torch.cat(all_vlad, dim=0)
    torch.save(vlad_db, str(db_path))
    names_path.write_text("\n".join(patch_names) + "\n")
    print(f"  DB saved: {db_path}  {vlad_db.shape}")


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


# ── Bag replay ────────────────────────────────────────────────────────────────

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
                  f"DINOv2 vs DINOv3 VLAD — {bag_name}")

    sum_path = bag_out / "summary.csv"
    with open(str(sum_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)

    return summaries


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DINOv2 vs DINOv3 VLAD for UAV geo-localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CFG)
    parser.add_argument("--build-codebooks", dest="build_codebooks", action="store_true",
                        help="Build DINOv2 aerial codebooks + patch DBs before benchmarking")
    parser.add_argument("--max-json-images", dest="max_json_images", type=int, default=10_000)
    parser.add_argument("--layers", nargs="+", type=int, default=[10, 11],
                        help="Layer indices to benchmark (e.g. --layers 10 11)")
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

    out_dir = PKG_DIR / "results" / "bench_dinov2_vs_dinov3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DB directories
    db_dirs = {}
    for layer in args.layers:
        db_dirs[f"dinov2_l{layer}"] = script_dir / "data" / f"descriptors_dinov2_vlad_l{layer}_aerial"
        db_dirs[f"dinov3_l{layer}"] = script_dir / "data" / f"descriptors_vlad_l{layer}_aerial"

    # ── build DINOv2 codebooks + patch DBs ─────────────────────────────
    if args.build_codebooks:
        print(f"\nSampling up to {args.max_json_images} images from {AERIAL_JSON.name} …")
        image_paths = _collect_image_paths(AERIAL_JSON, args.max_json_images)

        for layer in args.layers:
            key = f"dinov2_l{layer}"
            d   = db_dirs[key]
            cb  = d / "vlad_codebook.pt"
            if cb.exists():
                print(f"[skip] Codebook exists: {cb}")
            else:
                ext = DINOv2VLADExtractor(
                    codebook_path=None, layer_idx=layer,
                    image_size=DINOV2_IMAGE_SIZE, grayscale=GRAYSCALE, device="cuda",
                )
                build_aerial_codebook(d, layer, image_paths, ext)
                del ext
                torch.cuda.empty_cache()

            # Build patch DB
            db_pt = d / "vlad_descriptors.pt"
            if not db_pt.exists():
                ext = DINOv2VLADExtractor(
                    codebook_path=str(cb), layer_idx=layer,
                    image_size=DINOV2_IMAGE_SIZE, grayscale=GRAYSCALE, device="cuda",
                )
                build_patch_db(patches_dir, d, ext)
                del ext
                torch.cuda.empty_cache()

        # DINOv3 aerial DBs should already exist — build if missing
        for layer in args.layers:
            key = f"dinov3_l{layer}"
            d   = db_dirs[key]
            cb  = d / "vlad_codebook.pt"
            db_pt = d / "vlad_descriptors.pt"
            if not cb.exists():
                print(f"\n[{key}] Codebook missing — building …")
                ext = DINOv3VLADExtractor(
                    codebook_path=None, layer_idx=layer,
                    image_size=DINOV3_IMAGE_SIZE, grayscale=GRAYSCALE, device="cuda",
                )
                build_aerial_codebook(d, layer, image_paths, ext)
                del ext
                torch.cuda.empty_cache()
            if not db_pt.exists():
                print(f"\n[{key}] Patch DB missing — building …")
                ext = DINOv3VLADExtractor(
                    codebook_path=str(cb), layer_idx=layer,
                    image_size=DINOV3_IMAGE_SIZE, grayscale=GRAYSCALE, device="cuda",
                )
                build_patch_db(patches_dir, d, ext)
                del ext
                torch.cuda.empty_cache()

    # ── load methods ──────────────────────────────────────────────────────
    methods: List[Method] = []

    for mn, d in db_dirs.items():
        layer_idx = int(mn.split("_l")[1])
        db_pt   = d / "vlad_descriptors.pt"
        cb_pt   = d / "vlad_codebook.pt"
        n_path  = d / "patch_names.txt"
        if not db_pt.exists() or not cb_pt.exists():
            print(f"[warn] DB missing for {mn} at {d} — skipping")
            print(f"  Run with --build-codebooks to build it.")
            continue

        is_dinov2 = mn.startswith("dinov2")
        if is_dinov2:
            print(f"[{mn}] Loading DINOv2 (layer={layer_idx})")
            ext = DINOv2VLADExtractor(
                str(cb_pt), layer_idx, DINOV2_IMAGE_SIZE, GRAYSCALE, "cuda",
            )
        else:
            print(f"[{mn}] Loading DINOv3 (layer={layer_idx})")
            ext = DINOv3VLADExtractor(
                str(cb_pt), layer_idx, DINOV3_IMAGE_SIZE, GRAYSCALE, "cuda",
            )

        db, names = _load_db(db_pt, n_path)
        patch_enu = _patch_enu_array(names, gps_meta, enu)
        methods.append(Method(mn, ext, db, names, patch_enu))
        print(f"  DB: {db.shape}  codebook: K={ext.K}")

    if not methods:
        print("[error] No methods loaded.")
        return

    print(f"\nMethods: {[m.name for m in methods]}")

    # ── run bags ──────────────────────────────────────────────────────────
    all_summaries: Dict[str, List[dict]] = {}

    for bag_entry in BENCH_BAGS:
        sums = run_bag(bag_entry, methods, cfg, args.gt_radius, args.max_frames, out_dir)
        all_summaries[bag_entry["name"]] = sums

    # ── combined table ────────────────────────────────────────────────────
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

    # mean across bags
    print(f"\nMEAN across {len(BENCH_BAGS)} bags:")
    print(hdr); print(sep)
    method_names_all = [m.name for m in methods]
    for mn in method_names_all:
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

    # save combined CSV
    if combined_rows:
        f_out = out_dir / "combined_summary.csv"
        with open(str(f_out), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
            w.writeheader(); w.writerows(combined_rows)
        print(f"\n[done] Combined CSV: {f_out}")


if __name__ == "__main__":
    main()
