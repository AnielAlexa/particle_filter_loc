#!/usr/bin/env python3
"""bench_coarse_retrieval.py

Compare three coarse retrieval methods against RTK ground truth on an MCAP bag.

Methods
-------
  boq_trt   : DINOv3-S ViT + LoRA + BoQ  (TensorRT engine, 384-dim)
  vlad_l10  : DINOv3-S ViT + VLAD on layer-10 value features
  vlad_l11  : DINOv3-S ViT + VLAD on layer-11 value features

Per-frame metrics
-----------------
  error_m   : Distance (m) from top-1 patch centre to RTK fix
  cos_pos   : Cosine similarity to the GT patch (closest to RTK within --gt-radius)
  cos_neg   : Cosine similarity to the top-1 retrieved patch
  rank_gt   : Rank of GT patch in similarity order (1 = correct retrieval)

A frame is considered a correct retrieval (recall@1) when rank_gt == 1.

Usage
-----
  # Build layer-11 VLAD DB from satellite patches, then run bench:
  python3 bench_coarse_retrieval.py --build-vlad-l11

  # Build both VLAD DBs (if starting fresh):
  python3 bench_coarse_retrieval.py --build-vlad-l10 --build-vlad-l11

  # Run bench only (DBs already built):
  python3 bench_coarse_retrieval.py

  # Override bag path and start offset:
  python3 bench_coarse_retrieval.py --bag /path/to/bag.mcap --start-offset 40

  # Quick test with limited frames:
  python3 bench_coarse_retrieval.py --max-frames 200
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── project paths ──────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_DIR     = SCRIPTS_DIR.parent          # particle_filter_loc/
REPO_ROOT   = PKG_DIR.parent              # vpr_dinov3/

sys.path.insert(0, str(PKG_DIR))
from particle_filter_loc.geo_utils import ENUFrame, haversine_m
from particle_filter_loc.vlad_extractor import DINOv3VLADExtractor


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _import_match_module(script_dir: str):
    sdir = str(script_dir)
    if sdir not in sys.path:
        sys.path.insert(0, sdir)
    src = Path(script_dir) / "3_match_video_dinov3.py"
    spec = importlib.util.spec_from_file_location("match_video_dinov3", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_db(db_path: Path, names_path: Path) -> Tuple[torch.Tensor, List[str]]:
    data = torch.load(str(db_path), map_location="cuda")
    if isinstance(data, dict):
        db = data.get("descriptors", data.get("boq_descriptors", list(data.values())[0]))
    else:
        db = data
    db = F.normalize(db.float().cuda(), dim=1)
    with open(str(names_path)) as f:
        names = [l.strip() for l in f if l.strip()]
    assert len(names) == db.shape[0], \
        f"DB size mismatch: {db.shape[0]} descriptors vs {len(names)} names in {names_path}"
    return db, names


def _patch_enu_array(names: List[str], gps_meta: dict, enu: ENUFrame) -> np.ndarray:
    coords = []
    for name in names:
        m = gps_meta.get(name)
        if m:
            e, n = enu.wgs84_to_enu(m["lat"], m["lon"])
        else:
            e, n = 0.0, 0.0
        coords.append((e, n))
    return np.array(coords, dtype=np.float64)


# ── VLAD database builder ──────────────────────────────────────────────────────

def build_vlad_db(
    patches_dir: Path,
    out_dir: Path,
    layer_idx: int,
    image_size: int = 256,
    grayscale: bool = True,
    num_clusters: int = 16,
    batch_size: int = 8,
    max_kmeans_samples: int = 100_000,
) -> None:
    """Build VLAD codebook + descriptor database for a given layer index."""
    try:
        import fast_pytorch_kmeans as fpk
    except ImportError:
        raise ImportError(
            "fast_pytorch_kmeans is required to build VLAD databases.\n"
            "Install with:  pip install fast-pytorch-kmeans"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    codebook_path = out_dir / "vlad_codebook.pt"
    db_path       = out_dir / "vlad_descriptors.pt"
    names_path    = out_dir / "patch_names.txt"

    patch_paths = sorted(patches_dir.glob("*.png"))
    if not patch_paths:
        raise FileNotFoundError(f"No .png patches found in {patches_dir}")
    print(f"[build_vlad_l{layer_idx}] {len(patch_paths)} patches, "
          f"layer={layer_idx}, K={num_clusters}, grayscale={grayscale}")

    extractor = DINOv3VLADExtractor(
        codebook_path=None,
        layer_idx=layer_idx,
        image_size=image_size,
        grayscale=grayscale,
        device="cuda",
    )

    # ── feature extraction ────────────────────────────────────────────────
    from tqdm import tqdm

    all_feats: list[torch.Tensor] = []
    patch_names: list[str] = []

    for i in tqdm(range(0, len(patch_paths), batch_size),
                  desc=f"Extracting layer-{layer_idx} features"):
        batch = patch_paths[i : i + batch_size]
        tensors, names = [], []
        for p in batch:
            img = cv2.imread(str(p))
            if img is None:
                print(f"  [warn] Cannot read {p}, skipping")
                continue
            tensors.append(extractor.preprocess(img))
            names.append(p.stem)
        if not tensors:
            continue
        batch_t = torch.cat(tensors, dim=0)
        feats = extractor.extract_patch_features(batch_t)
        all_feats.append(feats.cpu())
        patch_names.extend(names)

    all_feats_t = torch.cat(all_feats, dim=0)   # [N, N_tokens, D]
    print(f"  features: {all_feats_t.shape}")

    # ── K-means codebook ──────────────────────────────────────────────────
    N, T, D = all_feats_t.shape
    flat = all_feats_t.reshape(-1, D)
    if flat.shape[0] > max_kmeans_samples:
        idx = torch.randperm(flat.shape[0])[:max_kmeans_samples]
        flat = flat[idx]
    print(f"  K-means: {flat.shape[0]} vectors, K={num_clusters}")
    flat = flat.cuda()
    kmeans = fpk.KMeans(n_clusters=num_clusters, mode="euclidean", verbose=1)
    kmeans.fit(flat)
    c_centers = kmeans.centroids.cpu().float()
    torch.save({"c_centers": c_centers}, str(codebook_path))
    print(f"  Codebook saved: {codebook_path}")

    # ── VLAD descriptors ──────────────────────────────────────────────────
    extractor._load_codebook(str(codebook_path))
    all_vlad = []
    for i in tqdm(range(len(patch_names)), desc="VLAD aggregation"):
        f_i = all_feats_t[i].unsqueeze(0).to(extractor.device)
        v_i = extractor._vlad_aggregate(f_i).cpu()
        all_vlad.append(v_i)
    vlad_db = torch.cat(all_vlad, dim=0)
    torch.save(vlad_db, str(db_path))
    names_path.write_text("\n".join(patch_names) + "\n")
    print(f"  DB saved: {db_path}  ({vlad_db.shape})")
    print(f"  Names saved: {names_path}  ({len(patch_names)} entries)")


# ── Game4Loc trained BoQ extractor (PyTorch) ──────────────────────────────────

class GameLoRABoQExtractor:
    """
    PyTorch DesModelDINOv3BoQ extractor.
    Loads a Game4Loc checkpoint (LoRA + BoQ, intermediate layer extraction).
    Interface matches the TRT and VLAD extractors: preprocess() + __call__().
    """

    def __init__(
        self,
        checkpoint_path: str,
        img_size: int = 256,
        grayscale: bool = True,
        num_queries: int = 16,
        mlp_output_dim: int = 384,
        lora_r: int = 2,
        lora_alpha: int = 4,
        lora_dropout: float = 0.2,
        intermediate_layer_idx: int = 11,
        intermediate_facet: str = "value",
        device: str = "cuda",
    ) -> None:
        import sys as _sys
        game4loc_root = str(REPO_ROOT / "GTA-UAV" / "Game4Loc")
        if game4loc_root not in _sys.path:
            _sys.path.insert(0, game4loc_root)
        from game4loc.models.model_dinov3_boq import DesModelDINOv3BoQ

        self.device = torch.device(device)
        self.img_size = img_size
        self.grayscale = grayscale

        print(f"[GameLoRABoQ] Building model (layer={intermediate_layer_idx}, "
              f"facet={intermediate_facet}, num_queries={num_queries}, lora_r={lora_r}) …")
        model = DesModelDINOv3BoQ(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            pretrained=True,
            img_size=img_size,
            share_weights=True,
            num_queries=num_queries,
            mlp_output_dim=mlp_output_dim,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_intermediate_layer=True,
            intermediate_layer_idx=intermediate_layer_idx,
            intermediate_facet=intermediate_facet,
        )

        print(f"[GameLoRABoQ] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [warn] Missing keys: {missing[:5]}{'…' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  [warn] Unexpected keys: {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")

        model.eval().to(self.device)
        self.model = model

        cfg = model.get_config()
        self.mean = torch.tensor(cfg["mean"], dtype=torch.float32,
                                 device=self.device).view(1, 3, 1, 1)
        self.std  = torch.tensor(cfg["std"],  dtype=torch.float32,
                                 device=self.device).view(1, 3, 1, 1)
        print("[GameLoRABoQ] Ready.")

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        frame = frame_bgr[y0:y0 + s, x0:x0 + s]
        frame = cv2.resize(frame, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_CUBIC)
        if self.grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.stack([gray, gray, gray], axis=2)
        else:
            frame = frame[:, :, ::-1]   # BGR → RGB
        tensor = (
            torch.from_numpy(frame.transpose(2, 0, 1).copy())
            .unsqueeze(0).float().to(self.device) / 255.0
        )
        return (tensor - self.mean) / self.std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            desc = self.model(img1=tensor)   # already L2-normalized [1, 384]
        return desc


def build_game_lora_db(
    patches_dir: Path,
    out_dir: Path,
    checkpoint_path: str,
    image_size: int = 256,
    grayscale: bool = True,
    **extractor_kwargs,
) -> None:
    """Extract Game4Loc BoQ descriptors for all satellite patches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path    = out_dir / "boq_descriptors.pt"
    names_path = out_dir / "patch_names.txt"

    patch_paths = sorted(patches_dir.glob("*.png"))
    if not patch_paths:
        raise FileNotFoundError(f"No .png patches found in {patches_dir}")
    print(f"[build_game_lora_db] {len(patch_paths)} patches")

    ext = GameLoRABoQExtractor(
        checkpoint_path=checkpoint_path,
        img_size=image_size,
        grayscale=grayscale,
        **extractor_kwargs,
    )

    from tqdm import tqdm

    all_descs: list[torch.Tensor] = []
    patch_names: list[str] = []

    for p in tqdm(patch_paths, desc="Game LoRA BoQ DB"):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [warn] Cannot read {p}, skipping")
            continue
        with torch.no_grad():
            tensor = ext.preprocess(img)
            desc   = F.normalize(ext(tensor).float(), dim=1)
        all_descs.append(desc.cpu())
        patch_names.append(p.stem)

    db = torch.cat(all_descs, dim=0)
    torch.save(db, str(db_path))
    names_path.write_text("\n".join(patch_names) + "\n")
    print(f"  DB saved: {db_path}  ({db.shape})")


# ── BoQ database builder ───────────────────────────────────────────────────────

def build_boq_db(
    patches_dir: Path,
    out_dir: Path,
    engine_path: Path,
    image_size: int = 256,
    grayscale: bool = True,
    batch_size: int = 1,
) -> None:
    """Extract BoQ descriptors for all satellite patches using a TRT engine."""
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path    = out_dir / "boq_descriptors.pt"
    names_path = out_dir / "patch_names.txt"

    patch_paths = sorted(patches_dir.glob("*.png"))
    if not patch_paths:
        raise FileNotFoundError(f"No .png patches found in {patches_dir}")

    engine_label = engine_path.stem
    print(f"[build_boq {engine_label}] {len(patch_paths)} patches")

    mod = _import_match_module(str(engine_path.parent))
    ext = mod.DINOv3LoRABoQExtractorTRT(
        engine_path=str(engine_path),
        image_size=image_size,
        grayscale=grayscale,
    )

    from tqdm import tqdm

    all_descs: list[torch.Tensor] = []
    patch_names: list[str] = []

    for p in tqdm(patch_paths, desc=f"BoQ {engine_label}"):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [warn] Cannot read {p}, skipping")
            continue
        with torch.no_grad():
            tensor = ext.preprocess(img)
            desc   = F.normalize(ext(tensor).float(), dim=1)
        all_descs.append(desc.cpu())
        patch_names.append(p.stem)

    db = torch.cat(all_descs, dim=0)   # [N, D]
    torch.save(db, str(db_path))
    names_path.write_text("\n".join(patch_names) + "\n")
    print(f"  DB saved: {db_path}  ({db.shape})")
    print(f"  Names saved: {names_path}")


# ── VLAD + PCA extractor ───────────────────────────────────────────────────────

class VLADPCAExtractor:
    """DINOv3VLADExtractor + pre-computed PCA projection.

    Loads pca_components.pt from the DB directory and applies it at query time.
    If pca_components.pt is absent, fits PCA on-the-fly from the full DB.
    """

    def __init__(
        self,
        base_extractor: "DINOv3VLADExtractor",
        db_full: torch.Tensor,       # [N, full_dim] — used to fit PCA if needed
        pca_path: Optional[Path] = None,
        pca_dim: int = 768,
        device: str = "cuda",
    ) -> None:
        self.base = base_extractor
        self.device = torch.device(device)

        if pca_path is not None and pca_path.exists():
            data = torch.load(str(pca_path), map_location=device, weights_only=False)
            self._mean = data["mean"].to(self.device)
            self._components = data["components"][:pca_dim].to(self.device)
            print(f"[VLADPCAExtractor] Loaded pre-computed PCA "
                  f"{db_full.shape[1]}→{self._components.shape[0]} dims")
        else:
            print(f"[VLADPCAExtractor] Fitting PCA on-the-fly ({db_full.shape}) → {pca_dim} dims")
            X = db_full.float().to(self.device)
            self._mean = X.mean(dim=0)
            _, _, Vh = torch.linalg.svd(X - self._mean, full_matrices=False)
            self._components = Vh[:pca_dim]
            print(f"  PCA fit done.")

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        return self.base.preprocess(frame_bgr)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        vlad = self.base(tensor).float().to(self.device)
        projected = (vlad - self._mean) @ self._components.T
        return F.normalize(projected, dim=1)

    def project_db(self, db_full: torch.Tensor) -> torch.Tensor:
        """Project full-dim DB to PCA space, L2-normalized."""
        X = db_full.float().to(self.device)
        return F.normalize((X - self._mean) @ self._components.T, dim=1)


# ── Method container ───────────────────────────────────────────────────────────

@dataclass
class Method:
    name: str
    extractor: object          # has .preprocess(frame_bgr) -> Tensor, __call__(Tensor) -> [1,D]
    db: torch.Tensor           # [N, D] float32 cuda L2-normalized
    patch_names: List[str]
    patch_enu: np.ndarray      # [N, 2] float64  (east, north)

    def retrieve(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """Returns (sims_np [N], top1_idx, top1_sim)."""
        tensor = self.extractor.preprocess(frame_bgr)
        with torch.no_grad():
            desc = F.normalize(self.extractor(tensor).float(), dim=1)
        sims = (desc @ self.db.T).squeeze(0).cpu().numpy()
        top1_idx = int(sims.argmax())
        return sims, top1_idx, float(sims[top1_idx])


# ── Metrics computation ────────────────────────────────────────────────────────

def compute_frame_metrics(
    method: Method,
    frame_bgr: np.ndarray,
    rtk_e: float,
    rtk_n: float,
    gt_patch_idx: int,          # -1 if no GT within radius
) -> dict:
    sims, top1_idx, top1_sim = method.retrieve(frame_bgr)

    # Top-1 retrieval error (always computed)
    top1_e, top1_n = method.patch_enu[top1_idx]
    error_m = math.hypot(top1_e - rtk_e, top1_n - rtk_n)

    # Recall@K
    top5_idx = np.argsort(sims)[::-1][:5]

    if gt_patch_idx >= 0:
        cos_pos  = float(sims[gt_patch_idx])
        cos_neg  = float(sims[top1_idx])   # sim to what the system actually returned
        rank_gt  = int((sims > sims[gt_patch_idx]).sum()) + 1
        r1 = int(top1_idx == gt_patch_idx)
        r5 = int(gt_patch_idx in top5_idx)
    else:
        cos_pos = cos_neg = float("nan")
        rank_gt = -1
        r1 = r5 = -1     # undefined

    return {
        "top1_patch": method.patch_names[top1_idx],
        "error_m":    error_m,
        "cos_pos":    cos_pos,
        "cos_neg":    cos_neg,
        "rank_gt":    rank_gt,
        "recall1":    r1,
        "recall5":    r5,
    }


# ── Summary stats ──────────────────────────────────────────────────────────────

def compute_summary(method_name: str, rows: List[dict]) -> dict:
    valid = [r for r in rows if r["recall1"] >= 0]
    errors = np.array([r["error_m"] for r in valid])
    cp = np.array([r["cos_pos"] for r in valid if not math.isnan(r["cos_pos"])])
    cn = np.array([r["cos_neg"] for r in valid if not math.isnan(r["cos_neg"])])
    r1 = np.array([r["recall1"] for r in valid])
    r5 = np.array([r["recall5"] for r in valid])

    return {
        "method":          method_name,
        "n_frames":        len(rows),
        "n_with_gt":       len(valid),
        "recall1_pct":     float(r1.mean() * 100) if len(r1) else float("nan"),
        "recall5_pct":     float(r5.mean() * 100) if len(r5) else float("nan"),
        "median_error_m":  float(np.median(errors)) if len(errors) else float("nan"),
        "p90_error_m":     float(np.percentile(errors, 90)) if len(errors) else float("nan"),
        "mean_cos_pos":    float(cp.mean()) if len(cp) else float("nan"),
        "mean_cos_neg":    float(cn.mean()) if len(cn) else float("nan"),
        "mean_margin":     float((cp - cn).mean()) if len(cp) else float("nan"),
    }


# ── Trajectory plot ────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "boq_trt":      "#2196F3",   # blue
    "boq_l9":       "#9C27B0",   # purple
    "game_lora":    "#E91E63",   # pink/red
    "game_lora_r4":    "#FF5722",   # deep orange  (lora_r=4, q8, mlp=512, l11)
    "game_lora_r2_l10": "#00BCD4",  # cyan         (lora_r=2, q16, mlp=384, l10)
    "game_lora_r2_l11": "#8BC34A",  # light green  (lora_r=2, q16, mlp=512, l11)
    "vlad_pca768":      "#795548",  # brown        (vlad l10 aerial + PCA 768)
    "vlad_l10":     "#4CAF50",   # green
    "vlad_l11":     "#FF9800",   # orange
}

def save_trajectory_plot(
    frame_records: List[dict],
    method_names: List[str],
    out_path: Path,
    bag_name: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ── collect series ────────────────────────────────────────────────────
    rtk_e  = np.array([r["rtk_e"]  for r in frame_records])
    rtk_n  = np.array([r["rtk_n"]  for r in frame_records])
    elapsed = np.array([r["elapsed_s"] for r in frame_records])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Coarse retrieval benchmark — {bag_name}", fontsize=13)

    ax_traj, ax_err, ax_cos = axes

    # ── left: trajectory ──────────────────────────────────────────────────
    ax_traj.scatter(rtk_e, rtk_n, c="black", s=4, label="RTK GT", zorder=5)
    for mn in method_names:
        color = METHOD_COLORS.get(mn, "gray")
        me_e = np.array([r[f"{mn}_top1_e"] for r in frame_records])
        me_n = np.array([r[f"{mn}_top1_n"] for r in frame_records])
        ax_traj.scatter(me_e, me_n, c=color, s=3, alpha=0.5, label=mn)

    ax_traj.set_xlabel("East (m)")
    ax_traj.set_ylabel("North (m)")
    ax_traj.set_title("Trajectory (ENU)")
    ax_traj.legend(loc="best", markerscale=3, fontsize=8)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, linewidth=0.4)

    # ── middle: error over time ───────────────────────────────────────────
    for mn in method_names:
        color = METHOD_COLORS.get(mn, "gray")
        err = np.array([r[f"{mn}_error_m"] for r in frame_records])
        ax_err.plot(elapsed, err, color=color, linewidth=0.8, alpha=0.8, label=mn)

    ax_err.set_xlabel("Elapsed (s)")
    ax_err.set_ylabel("Top-1 error (m)")
    ax_err.set_title("Retrieval error vs time")
    ax_err.legend(loc="upper right", fontsize=8)
    ax_err.grid(True, linewidth=0.4)
    ax_err.set_ylim(bottom=0)

    # ── right: cosine similarity over time ────────────────────────────────
    for mn in method_names:
        color = METHOD_COLORS.get(mn, "gray")
        cp = np.array([r[f"{mn}_cos_pos"] for r in frame_records], dtype=float)
        cn = np.array([r[f"{mn}_cos_neg"] for r in frame_records], dtype=float)
        # replace nan with nan (keep gaps in plot)
        cp[np.isnan(cp)] = np.nan
        cn[np.isnan(cn)] = np.nan
        ax_cos.plot(elapsed, cp, color=color, linewidth=0.9, label=f"{mn} pos")
        ax_cos.plot(elapsed, cn, color=color, linewidth=0.9, linestyle="--",
                    alpha=0.6, label=f"{mn} neg")

    ax_cos.set_xlabel("Elapsed (s)")
    ax_cos.set_ylabel("Cosine similarity")
    ax_cos.set_title("cos_pos (solid) / cos_neg (dashed)")
    ax_cos.legend(loc="lower right", fontsize=7)
    ax_cos.grid(True, linewidth=0.4)
    ax_cos.set_ylim(0, 1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


# ── Main bench loop ────────────────────────────────────────────────────────────

def run_bench(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    mcfg = cfg["matchers"]
    rcfg = cfg["replay"]

    script_dir  = Path(mcfg["script_dir"])
    patches_dir = script_dir / mcfg.get("patches_dir", "data/patches")
    image_size  = int(mcfg.get("image_size", 256))
    grayscale   = bool(mcfg.get("grayscale", True))
    num_clusters = int(mcfg.get("vlad_num_clusters", 16))

    # VLAD DB paths (layer-specific) — use aerial codebook variants
    vlad_l10_dir = script_dir / "data" / "descriptors_vlad_l10_aerial"
    vlad_l11_dir = script_dir / "data" / "descriptors_vlad_l11_aerial"

    # BoQ l9 engine + DB paths
    boq_l9_engine = script_dir / "dinov3_lora_gray_boq_l9_value_256.engine"
    boq_l9_dir    = script_dir / "data" / "descriptors_boq_l9"

    # Game4Loc trained BoQ DB path
    game_ckpt_path = args.game_lora_ckpt
    game_lora_dir  = script_dir / "data" / "descriptors_game_lora_boq_l11"

    # Game4Loc lora_r=4 / num_queries=8 / mlp=512 / layer=11 model
    game_r4_ckpt_path = args.game_lora_r4_ckpt
    game_lora_r4_dir  = script_dir / "data" / "descriptors_game_lora_r4_q8_l11"

    # Game4Loc lora_r=2 / num_queries=16 / mlp=384 / layer=10 model
    game_r2_l10_ckpt_path = args.game_lora_r2_l10_ckpt
    game_lora_r2_l10_dir  = script_dir / "data" / "descriptors_game_lora_r2_l10_q16"

    # Game4Loc lora_r=2 / num_queries=16 / mlp=512 / layer=11 model
    game_r2_l11_ckpt_path = args.game_lora_r2_l11_ckpt
    game_lora_r2_l11_dir  = script_dir / "data" / "descriptors_game_lora_r2_l11_q16"

    # ── optional DB builds ────────────────────────────────────────────────
    if args.build_game_lora:
        if not game_ckpt_path:
            raise ValueError("--build-game-lora requires --game-lora-ckpt <path>")
        build_game_lora_db(patches_dir, game_lora_dir,
                           checkpoint_path=game_ckpt_path,
                           image_size=image_size, grayscale=grayscale)

    if args.build_game_lora_r4:
        if not game_r4_ckpt_path:
            raise ValueError("--build-game-lora-r4 requires --game-lora-r4-ckpt <path>")
        build_game_lora_db(patches_dir, game_lora_r4_dir,
                           checkpoint_path=game_r4_ckpt_path,
                           image_size=image_size, grayscale=grayscale,
                           num_queries=8, mlp_output_dim=512,
                           lora_r=4, lora_alpha=8,
                           intermediate_layer_idx=11, intermediate_facet="value")

    if args.build_game_lora_r2_l10:
        if not game_r2_l10_ckpt_path:
            raise ValueError("--build-game-lora-r2-l10 requires --game-lora-r2-l10-ckpt <path>")
        build_game_lora_db(patches_dir, game_lora_r2_l10_dir,
                           checkpoint_path=game_r2_l10_ckpt_path,
                           image_size=image_size, grayscale=grayscale,
                           num_queries=16, mlp_output_dim=384,
                           lora_r=2, lora_alpha=4,
                           intermediate_layer_idx=10, intermediate_facet="value")

    if args.build_game_lora_r2_l11:
        if not game_r2_l11_ckpt_path:
            raise ValueError("--build-game-lora-r2-l11 requires --game-lora-r2-l11-ckpt <path>")
        build_game_lora_db(patches_dir, game_lora_r2_l11_dir,
                           checkpoint_path=game_r2_l11_ckpt_path,
                           image_size=image_size, grayscale=grayscale,
                           num_queries=16, mlp_output_dim=512,
                           lora_r=2, lora_alpha=4,
                           intermediate_layer_idx=11, intermediate_facet="value")

    if args.build_boq_l9:
        if not boq_l9_engine.exists():
            raise FileNotFoundError(f"BoQ l9 engine not found: {boq_l9_engine}")
        build_boq_db(patches_dir, boq_l9_dir, boq_l9_engine,
                     image_size=image_size, grayscale=grayscale)

    if args.build_vlad_l10:
        build_vlad_db(patches_dir, vlad_l10_dir, layer_idx=10,
                      image_size=image_size, grayscale=grayscale,
                      num_clusters=num_clusters)

    if args.build_vlad_l11:
        build_vlad_db(patches_dir, vlad_l11_dir, layer_idx=11,
                      image_size=image_size, grayscale=grayscale,
                      num_clusters=num_clusters)

    # ── load methods ──────────────────────────────────────────────────────
    methods: List[Method] = []

    # GPS metadata (shared)
    meta_path = script_dir / mcfg.get("gps_metadata_path", "data/patches/gps_metadata.json")
    with open(str(meta_path)) as f:
        gps_meta: Dict = json.load(f)

    enu_orig = cfg["enu_origin"]
    enu = ENUFrame(enu_orig["lat"], enu_orig["lon"])

    # 1. BoQ TRT
    boq_engine = script_dir / mcfg["boq_engine_path"]
    boq_db_pt  = script_dir / mcfg["boq_database_path"]
    boq_names  = script_dir / mcfg["patch_names_path"]

    if not boq_engine.exists():
        print(f"[warn] BoQ TRT engine not found: {boq_engine}  — skipping boq_trt method")
    elif not boq_db_pt.exists():
        print(f"[warn] BoQ DB not found: {boq_db_pt}  — skipping boq_trt method")
    else:
        print(f"[boq_trt] Loading TRT engine: {boq_engine}")
        mod = _import_match_module(str(script_dir))
        boq_ext = mod.DINOv3LoRABoQExtractorTRT(
            engine_path=str(boq_engine),
            image_size=image_size,
            grayscale=grayscale,
        )
        db, names = _load_db(boq_db_pt, boq_names)
        patch_enu = _patch_enu_array(names, gps_meta, enu)
        methods.append(Method("boq_trt", boq_ext, db, names, patch_enu))
        print(f"  DB: {db.shape}  ({len(names)} patches)")

    # 2. BoQ TRT l9
    boq_l9_db_pt  = boq_l9_dir / "boq_descriptors.pt"
    boq_l9_names  = boq_l9_dir / "patch_names.txt"

    if not boq_l9_engine.exists():
        print(f"[warn] BoQ l9 engine not found: {boq_l9_engine}  — skipping boq_l9 method")
    elif not boq_l9_db_pt.exists():
        print(f"[warn] BoQ l9 DB not found: {boq_l9_db_pt}  — skipping boq_l9 method")
        print("  Run with --build-boq-l9 to build it first.")
    else:
        print(f"[boq_l9] Loading TRT engine: {boq_l9_engine}")
        mod_l9 = _import_match_module(str(script_dir))
        boq_l9_ext = mod_l9.DINOv3LoRABoQExtractorTRT(
            engine_path=str(boq_l9_engine),
            image_size=image_size,
            grayscale=grayscale,
        )
        db_l9, names_l9 = _load_db(boq_l9_db_pt, boq_l9_names)
        patch_enu_l9 = _patch_enu_array(names_l9, gps_meta, enu)
        methods.append(Method("boq_l9", boq_l9_ext, db_l9, names_l9, patch_enu_l9))
        print(f"  DB: {db_l9.shape}  ({len(names_l9)} patches)")

    # 3. Game4Loc trained BoQ (PyTorch, layer 11 value, LoRA r=2)
    game_db_pt    = game_lora_dir / "boq_descriptors.pt"
    game_names_pt = game_lora_dir / "patch_names.txt"

    if not game_ckpt_path:
        print("[warn] --game-lora-ckpt not provided — skipping game_lora method")
    elif not game_db_pt.exists():
        print(f"[warn] Game LoRA BoQ DB not found: {game_db_pt}  — skipping game_lora method")
        print("  Run with --build-game-lora --game-lora-ckpt <path> to build it first.")
    else:
        print(f"[game_lora] Loading Game4Loc BoQ from checkpoint: {game_ckpt_path}")
        game_ext = GameLoRABoQExtractor(
            checkpoint_path=game_ckpt_path,
            img_size=image_size,
            grayscale=grayscale,
        )
        db_g, names_g = _load_db(game_db_pt, game_names_pt)
        patch_enu_g = _patch_enu_array(names_g, gps_meta, enu)
        methods.append(Method("game_lora", game_ext, db_g, names_g, patch_enu_g))
        print(f"  DB: {db_g.shape}  ({len(names_g)} patches)")

    # 4. Game4Loc trained BoQ lora_r=4 (PyTorch, layer 11 value, q=8, mlp=512)
    game_r4_db_pt    = game_lora_r4_dir / "boq_descriptors.pt"
    game_r4_names_pt = game_lora_r4_dir / "patch_names.txt"

    if not game_r4_ckpt_path:
        print("[warn] --game-lora-r4-ckpt not provided — skipping game_lora_r4 method")
    elif not game_r4_db_pt.exists():
        print(f"[warn] Game LoRA r4 BoQ DB not found: {game_r4_db_pt}  — skipping game_lora_r4 method")
        print("  Run with --build-game-lora-r4 --game-lora-r4-ckpt <path> to build it first.")
    else:
        print(f"[game_lora_r4] Loading Game4Loc r4 BoQ from checkpoint: {game_r4_ckpt_path}")
        game_r4_ext = GameLoRABoQExtractor(
            checkpoint_path=game_r4_ckpt_path,
            img_size=image_size,
            grayscale=grayscale,
            num_queries=8,
            mlp_output_dim=512,
            lora_r=4,
            lora_alpha=8,
            intermediate_layer_idx=11,
            intermediate_facet="value",
        )
        db_r4, names_r4 = _load_db(game_r4_db_pt, game_r4_names_pt)
        patch_enu_r4 = _patch_enu_array(names_r4, gps_meta, enu)
        methods.append(Method("game_lora_r4", game_r4_ext, db_r4, names_r4, patch_enu_r4))
        print(f"  DB: {db_r4.shape}  ({len(names_r4)} patches)")

    # 5. Game4Loc trained BoQ lora_r=2 (PyTorch, layer 10 value, q=16, mlp=384)
    game_r2_l10_db_pt    = game_lora_r2_l10_dir / "boq_descriptors.pt"
    game_r2_l10_names_pt = game_lora_r2_l10_dir / "patch_names.txt"

    if not game_r2_l10_ckpt_path:
        print("[warn] --game-lora-r2-l10-ckpt not provided — skipping game_lora_r2_l10 method")
    elif not game_r2_l10_db_pt.exists():
        print(f"[warn] Game LoRA r2 l10 DB not found: {game_r2_l10_db_pt}  — skipping game_lora_r2_l10 method")
        print("  Run with --build-game-lora-r2-l10 --game-lora-r2-l10-ckpt <path> to build it first.")
    else:
        print(f"[game_lora_r2_l10] Loading from checkpoint: {game_r2_l10_ckpt_path}")
        game_r2_l10_ext = GameLoRABoQExtractor(
            checkpoint_path=game_r2_l10_ckpt_path,
            img_size=image_size,
            grayscale=grayscale,
            num_queries=16,
            mlp_output_dim=384,
            lora_r=2,
            lora_alpha=4,
            intermediate_layer_idx=10,
            intermediate_facet="value",
        )
        db_r2, names_r2 = _load_db(game_r2_l10_db_pt, game_r2_l10_names_pt)
        patch_enu_r2 = _patch_enu_array(names_r2, gps_meta, enu)
        methods.append(Method("game_lora_r2_l10", game_r2_l10_ext, db_r2, names_r2, patch_enu_r2))
        print(f"  DB: {db_r2.shape}  ({len(names_r2)} patches)")

    # 6 & 7. VLAD methods (share model weights, different layer hooks + codebooks)
    vlad_configs = [
        ("vlad_l10", 10, vlad_l10_dir),
        ("vlad_l11", 11, vlad_l11_dir),
    ]

    for mname, layer_idx, vlad_dir in vlad_configs:
        db_pt  = vlad_dir / "vlad_descriptors.pt"
        cb_pt  = vlad_dir / "vlad_codebook.pt"
        n_path = vlad_dir / "patch_names.txt"
        if not db_pt.exists() or not cb_pt.exists():
            print(f"[warn] VLAD DB missing for {mname} at {vlad_dir}  — skipping")
            print(f"  Run with --build-vlad-l{layer_idx} to build it first.")
            continue
        print(f"[{mname}] Loading VLAD extractor (layer={layer_idx})")
        ext = DINOv3VLADExtractor(
            codebook_path=str(cb_pt),
            layer_idx=layer_idx,
            image_size=image_size,
            grayscale=grayscale,
            device="cuda",
        )
        db, names = _load_db(db_pt, n_path)
        patch_enu = _patch_enu_array(names, gps_meta, enu)
        methods.append(Method(mname, ext, db, names, patch_enu))
        print(f"  DB: {db.shape}  ({len(names)} patches)")

    # 6. Game4Loc trained BoQ lora_r=2 (PyTorch, layer 11 value, q=16, mlp=512)
    game_r2_l11_db_pt    = game_lora_r2_l11_dir / "boq_descriptors.pt"
    game_r2_l11_names_pt = game_lora_r2_l11_dir / "patch_names.txt"

    if not game_r2_l11_ckpt_path:
        print("[warn] --game-lora-r2-l11-ckpt not provided — skipping game_lora_r2_l11 method")
    elif not game_r2_l11_db_pt.exists():
        print(f"[warn] Game LoRA r2 l11 DB not found: {game_r2_l11_db_pt}  — skipping game_lora_r2_l11 method")
        print("  Run with --build-game-lora-r2-l11 --game-lora-r2-l11-ckpt <path> to build it first.")
    else:
        print(f"[game_lora_r2_l11] Loading from checkpoint: {game_r2_l11_ckpt_path}")
        game_r2_l11_ext = GameLoRABoQExtractor(
            checkpoint_path=game_r2_l11_ckpt_path,
            img_size=image_size,
            grayscale=grayscale,
            num_queries=16,
            mlp_output_dim=512,
            lora_r=2,
            lora_alpha=4,
            intermediate_layer_idx=11,
            intermediate_facet="value",
        )
        db_r2_l11, names_r2_l11 = _load_db(game_r2_l11_db_pt, game_r2_l11_names_pt)
        patch_enu_r2_l11 = _patch_enu_array(names_r2_l11, gps_meta, enu)
        methods.append(Method("game_lora_r2_l11", game_r2_l11_ext, db_r2_l11, names_r2_l11, patch_enu_r2_l11))
        print(f"  DB: {db_r2_l11.shape}  ({len(names_r2_l11)} patches)")

    # vlad_pca768 — same vlad_l10_aerial DB projected to 768 dims via pre-computed PCA
    vlad_pca768_full_pt = vlad_l10_dir / "vlad_descriptors.pt"
    vlad_pca768_pca_pt  = vlad_l10_dir / "pca_components.pt"
    vlad_pca768_names   = vlad_l10_dir / "patch_names.txt"

    if not vlad_pca768_full_pt.exists():
        print(f"[warn] vlad_l10_aerial DB not found — skipping vlad_pca768")
    else:
        print(f"[vlad_pca768] Loading vlad_l10_aerial + PCA→768")
        base_vlad_ext = DINOv3VLADExtractor(
            codebook_path=str(vlad_l10_dir / "vlad_codebook.pt"),
            layer_idx=10,
            image_size=image_size,
            grayscale=grayscale,
            device="cuda",
        )
        db_full_pca, names_pca = _load_db(vlad_pca768_full_pt, vlad_pca768_names)
        # reload unormalized for PCA fitting
        raw = torch.load(str(vlad_pca768_full_pt), map_location="cuda", weights_only=False)
        if isinstance(raw, dict):
            raw = raw.get("descriptors", list(raw.values())[0])
        raw = raw.float().cuda()
        pca_ext = VLADPCAExtractor(
            base_extractor=base_vlad_ext,
            db_full=raw,
            pca_path=vlad_pca768_pca_pt,
            pca_dim=768,
        )
        db_pca768 = pca_ext.project_db(raw)
        patch_enu_pca = _patch_enu_array(names_pca, gps_meta, enu)
        methods.append(Method("vlad_pca768", pca_ext, db_pca768, names_pca, patch_enu_pca))
        print(f"  DB: {db_pca768.shape}  ({len(names_pca)} patches)")

    if not methods:
        print("[error] No methods could be loaded. Aborting.")
        return

    method_names = [m.name for m in methods]
    print(f"\nRunning bench with methods: {method_names}")

    # ── MCAP setup ────────────────────────────────────────────────────────
    mcap_path  = args.bag or rcfg["mcap_path"]
    start_off  = args.start_offset if args.start_offset is not None \
                 else rcfg.get("start_offset_s", 0.0)
    alt_min    = rcfg.get("altitude_min_process_m", 0.0)
    cam_topic  = rcfg.get("camera_topic", "/camera/image_mono")
    rtk_topic  = rcfg.get("rtk_topic", "/m300/rtk/fix")
    alt_topic  = rcfg.get("altimeter_topic", "/altimeter/range")
    subsample  = args.subsample if args.subsample is not None else rcfg.get("camera_subsample", 1)

    bag_name   = args.bag_name or Path(mcap_path).stem.replace("_0", "")
    out_dir    = PKG_DIR / "results" / "bench_coarse" / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv    = out_dir / "per_frame.csv"
    out_sum    = out_dir / "summary.csv"
    out_plot   = out_dir / "trajectory.png"

    gt_radius_m = args.gt_radius

    print(f"[mcap] Opening: {mcap_path}")
    print(f"[mcap] start_offset={start_off}s  alt_min={alt_min}m  gt_radius={gt_radius_m}m")

    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    from collections import deque

    # ── per-frame accumulation ────────────────────────────────────────────
    frame_records: List[dict] = []      # for trajectory plot
    per_method_rows: Dict[str, List[dict]] = {m.name: [] for m in methods}

    # CSV fieldnames
    base_fields = ["frame_idx", "elapsed_s", "timestamp_ns",
                   "rtk_lat", "rtk_lon", "rtk_e", "rtk_n",
                   "gt_patch", "has_gt", "altitude_m"]
    method_fields = []
    for mn in method_names:
        method_fields += [
            f"{mn}_top1_patch", f"{mn}_error_m",
            f"{mn}_cos_pos", f"{mn}_cos_neg",
            f"{mn}_rank_gt", f"{mn}_recall1", f"{mn}_recall5",
        ]
    all_fields = base_fields + method_fields

    # State
    gt_lat, gt_lon         = None, None
    altitude_buf           = deque(maxlen=10)
    current_altitude_m     = 0.0
    t_start                = None
    camera_frame_idx       = 0

    with open(str(out_csv), "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=all_fields)
        writer.writeheader()

        with open(str(mcap_path), "rb") as bag_f:
            reader = make_reader(bag_f, decoder_factories=[DecoderFactory()])
            topics = [cam_topic, rtk_topic, alt_topic]

            for schema, channel, message, decoded_msg in reader.iter_decoded_messages(topics=topics):
                topic  = channel.topic
                ts_ns  = message.log_time

                if t_start is None:
                    t_start = ts_ns
                elapsed_s = (ts_ns - t_start) * 1e-9

                if elapsed_s < start_off:
                    continue

                # altimeter
                if topic == alt_topic:
                    altitude_buf.append(float(decoded_msg.range))
                    current_altitude_m = float(np.median(altitude_buf))
                    continue

                # RTK
                if topic == rtk_topic:
                    gt_lat = decoded_msg.latitude
                    gt_lon = decoded_msg.longitude
                    continue

                # Camera
                if topic != cam_topic:
                    continue

                camera_frame_idx += 1
                if camera_frame_idx % subsample != 0:
                    continue
                if args.max_frames and camera_frame_idx > args.max_frames:
                    print(f"[mcap] Reached max_frames={args.max_frames}, stopping.")
                    break

                # altitude gate
                if alt_min > 0 and current_altitude_m < alt_min:
                    continue

                # decode image
                h = decoded_msg.height
                w = decoded_msg.width
                if decoded_msg.encoding == "mono8":
                    gray = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(h, w)
                    frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                elif decoded_msg.encoding in ("bgr8", "rgb8"):
                    frame_bgr = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(h, w, 3)
                    if decoded_msg.encoding == "rgb8":
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                else:
                    continue

                # skip frames with no RTK yet
                if gt_lat is None:
                    continue

                rtk_e, rtk_n = enu.wgs84_to_enu(gt_lat, gt_lon)

                # find GT patch (closest within gt_radius_m using first method's ENU table)
                # all methods share the same patch set
                ref_method = methods[0]
                dx = ref_method.patch_enu[:, 0] - rtk_e
                dy = ref_method.patch_enu[:, 1] - rtk_n
                dists = np.hypot(dx, dy)
                gt_idx_ref = int(dists.argmin())
                has_gt = dists[gt_idx_ref] <= gt_radius_m
                gt_patch_name = ref_method.patch_names[gt_idx_ref] if has_gt else ""

                row: dict = {
                    "frame_idx":    camera_frame_idx,
                    "elapsed_s":    round(elapsed_s, 3),
                    "timestamp_ns": ts_ns,
                    "rtk_lat":      round(gt_lat, 8),
                    "rtk_lon":      round(gt_lon, 8),
                    "rtk_e":        round(rtk_e, 2),
                    "rtk_n":        round(rtk_n, 2),
                    "gt_patch":     gt_patch_name,
                    "has_gt":       int(has_gt),
                    "altitude_m":   round(current_altitude_m, 1),
                }

                traj_row: dict = {
                    "elapsed_s": elapsed_s,
                    "rtk_e": rtk_e, "rtk_n": rtk_n,
                }

                t0 = time.monotonic()
                for method in methods:
                    # find GT index in this method's patch list (same names, same order)
                    if has_gt:
                        try:
                            gt_idx = method.patch_names.index(gt_patch_name)
                        except ValueError:
                            gt_idx = -1
                    else:
                        gt_idx = -1

                    metrics = compute_frame_metrics(method, frame_bgr, rtk_e, rtk_n, gt_idx)
                    mn = method.name
                    row[f"{mn}_top1_patch"] = metrics["top1_patch"]
                    row[f"{mn}_error_m"]    = round(metrics["error_m"], 2)
                    row[f"{mn}_cos_pos"]    = round(metrics["cos_pos"], 4) if not math.isnan(metrics["cos_pos"]) else ""
                    row[f"{mn}_cos_neg"]    = round(metrics["cos_neg"], 4)
                    row[f"{mn}_rank_gt"]    = metrics["rank_gt"]
                    row[f"{mn}_recall1"]    = metrics["recall1"]
                    row[f"{mn}_recall5"]    = metrics["recall5"]

                    per_method_rows[mn].append(metrics)

                    # for trajectory plot
                    top1_idx_for_plot = method.patch_names.index(metrics["top1_patch"])
                    traj_row[f"{mn}_top1_e"]   = method.patch_enu[top1_idx_for_plot, 0]
                    traj_row[f"{mn}_top1_n"]   = method.patch_enu[top1_idx_for_plot, 1]
                    traj_row[f"{mn}_error_m"]  = metrics["error_m"]
                    traj_row[f"{mn}_cos_pos"]  = metrics["cos_pos"]
                    traj_row[f"{mn}_cos_neg"]  = metrics["cos_neg"]

                dt_ms = (time.monotonic() - t0) * 1000
                frame_records.append(traj_row)
                writer.writerow(row)

                if camera_frame_idx % 50 == 1:
                    err_str = "  ".join(
                        f"{m.name}:{per_method_rows[m.name][-1]['error_m']:.0f}m"
                        for m in methods
                    )
                    print(f"  frame {camera_frame_idx:4d}  t={elapsed_s:.1f}s  "
                          f"alt={current_altitude_m:.0f}m  gt={'Y' if has_gt else 'N'}  "
                          f"{err_str}  [{dt_ms:.0f}ms]")

    print(f"\n[bench] Per-frame CSV: {out_csv}  ({len(frame_records)} frames)")

    # ── summary ───────────────────────────────────────────────────────────
    summaries = []
    for mn, rows in per_method_rows.items():
        s = compute_summary(mn, rows)
        summaries.append(s)

    _print_summary_table(summaries)

    sum_fields = list(summaries[0].keys()) if summaries else []
    with open(str(out_sum), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summaries)
    print(f"[bench] Summary CSV: {out_sum}")

    # ── trajectory plot ───────────────────────────────────────────────────
    if frame_records:
        save_trajectory_plot(frame_records, method_names, out_plot, bag_name)

    return summaries


# ── Summary helpers ───────────────────────────────────────────────────────────

def _print_summary_table(summaries: List[dict], prefix: str = "") -> None:
    header = (f"{'method':<12} {'n_gt':>6} {'R@1%':>6} {'R@5%':>6} "
              f"{'med_err':>8} {'p90_err':>8} {'cos+':>6} {'cos-':>6} {'margin':>7}")
    sep = "─" * len(header)
    if prefix:
        print(f"\n{prefix}")
    print(f"\n{sep}")
    print(header)
    print(sep)
    for s in summaries:
        print(
            f"{s['method']:<12} {s['n_with_gt']:>6d} "
            f"{s['recall1_pct']:>5.1f}% {s['recall5_pct']:>5.1f}%  "
            f"{s['median_error_m']:>7.1f}m {s['p90_error_m']:>7.1f}m  "
            f"{s['mean_cos_pos']:>6.3f} {s['mean_cos_neg']:>6.3f} {s['mean_margin']:>+7.3f}"
        )
    print(sep)


def run_all_bags(args: argparse.Namespace) -> None:
    bags_yaml = yaml.safe_load(open(args.bags))
    all_results: Dict[str, List[dict]] = {}   # bag_name -> summaries list

    for entry in bags_yaml["bags"]:
        bag_name  = entry["name"]
        mcap_path = entry["mcap_path"]
        start_off = entry.get("start_offset_s", 0.0)
        notes     = entry.get("notes", "")

        print(f"\n{'='*70}")
        print(f"[multi-bench] {bag_name}  —  {Path(mcap_path).name}")
        if notes:
            print(f"  notes: {notes}")
        print(f"{'='*70}")

        # Clone args and override per-bag fields
        import copy
        bag_args = copy.copy(args)
        bag_args.bag          = mcap_path
        bag_args.bag_name     = bag_name
        bag_args.start_offset = start_off
        # Never rebuild DBs after the first bag
        bag_args.build_game_lora = False
        bag_args.build_boq_l9   = False
        bag_args.build_vlad_l10 = False
        bag_args.build_vlad_l11 = False

        try:
            summaries = run_bench(bag_args)
            all_results[bag_name] = summaries
        except Exception as exc:
            import traceback
            print(f"[multi-bench] ERROR on {bag_name}: {exc}")
            traceback.print_exc()
            all_results[bag_name] = []

    # ── Cross-bag combined table ──────────────────────────────────────────
    # One row per (bag, method)
    print(f"\n\n{'='*70}")
    print("COMBINED SUMMARY — all bags")
    print(f"{'='*70}")

    combined_header = (
        f"{'bag':<14} {'method':<12} {'R@1%':>6} {'R@5%':>6} "
        f"{'med_err':>8} {'p90_err':>8} {'cos+':>6} {'cos-':>6} {'margin':>7}"
    )
    sep = "─" * len(combined_header)
    print(combined_header)
    print(sep)

    combined_rows = []
    for bag_name, summaries in all_results.items():
        for s in summaries:
            print(
                f"{bag_name:<14} {s['method']:<12} "
                f"{s['recall1_pct']:>5.1f}% {s['recall5_pct']:>5.1f}%  "
                f"{s['median_error_m']:>7.1f}m {s['p90_error_m']:>7.1f}m  "
                f"{s['mean_cos_pos']:>6.3f} {s['mean_cos_neg']:>6.3f} {s['mean_margin']:>+7.3f}"
            )
            combined_rows.append({"bag": bag_name, **s})
        if summaries:
            print(sep)

    # Save combined CSV
    out_dir = PKG_DIR / "results" / "bench_coarse"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_combined = out_dir / "combined_summary.csv"
    if combined_rows:
        fields = list(combined_rows[0].keys())
        with open(str(out_combined), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(combined_rows)
        print(f"\n[multi-bench] Combined CSV: {out_combined}")

    # Per-method aggregate across bags
    print(f"\n{'='*70}")
    print("MEAN ACROSS ALL BAGS (per method)")
    print(f"{'='*70}")
    method_names_all = list({s["method"] for rows in all_results.values() for s in rows})
    print(combined_header)
    print(sep)
    for mn in sorted(method_names_all):
        mn_rows = [s for rows in all_results.values() for s in rows if s["method"] == mn]
        if not mn_rows:
            continue
        r1  = np.nanmean([r["recall1_pct"]    for r in mn_rows])
        r5  = np.nanmean([r["recall5_pct"]    for r in mn_rows])
        med = np.nanmean([r["median_error_m"] for r in mn_rows])
        p90 = np.nanmean([r["p90_error_m"]    for r in mn_rows])
        cp  = np.nanmean([r["mean_cos_pos"]   for r in mn_rows])
        cn  = np.nanmean([r["mean_cos_neg"]   for r in mn_rows])
        mg  = np.nanmean([r["mean_margin"]    for r in mn_rows])
        print(
            f"{'MEAN':<14} {mn:<12} "
            f"{r1:>5.1f}% {r5:>5.1f}%  "
            f"{med:>7.1f}m {p90:>7.1f}m  "
            f"{cp:>6.3f} {cn:>6.3f} {mg:>+7.3f}"
        )
    print(sep)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    default_cfg = str(PKG_DIR / "config" / "pf_config.yaml")

    parser = argparse.ArgumentParser(
        description="Benchmark coarse retrieval methods vs RTK ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=default_cfg,
                        help="Path to pf_config.yaml")
    parser.add_argument("--bags", default=None,
                        help="Path to bags.yaml — run all bags sequentially and print combined summary")
    parser.add_argument("--bag",  default=None,
                        help="Single MCAP bag path (overrides config)")
    parser.add_argument("--bag-name", dest="bag_name", default=None,
                        help="Override bag name used in output paths")
    parser.add_argument("--start-offset", dest="start_offset", type=float, default=None,
                        help="Skip this many seconds from bag start")
    parser.add_argument("--gt-radius", dest="gt_radius", type=float, default=50.0,
                        help="Radius (m) within which a patch is considered GT")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None,
                        help="Stop after this many camera frames (for debugging)")
    parser.add_argument("--subsample", dest="subsample", type=int, default=None,
                        help="Process every Nth camera frame (e.g. 5 = 5x faster)")
    parser.add_argument("--build-game-lora", dest="build_game_lora", action="store_true",
                        help="Build Game4Loc LoRA BoQ DB from the trained checkpoint before benchmarking")
    parser.add_argument("--game-lora-ckpt", dest="game_lora_ckpt", default=None,
                        help="Path to Game4Loc weights_end.pth checkpoint")
    parser.add_argument("--build-game-lora-r4", dest="build_game_lora_r4", action="store_true",
                        help="Build game_lora_r4 DB (lora_r=4, q8, mlp=512, l11) before benchmarking")
    parser.add_argument("--game-lora-r4-ckpt", dest="game_lora_r4_ckpt", default=None,
                        help="Path to weights_end_lora4_q8.pth checkpoint")
    parser.add_argument("--build-game-lora-r2-l10", dest="build_game_lora_r2_l10", action="store_true",
                        help="Build game_lora_r2_l10 DB (lora_r=2, q16, mlp=384, l10) before benchmarking")
    parser.add_argument("--game-lora-r2-l10-ckpt", dest="game_lora_r2_l10_ckpt", default=None,
                        help="Path to weights_end_lora2_l10_q16.pth checkpoint")
    parser.add_argument("--build-game-lora-r2-l11", dest="build_game_lora_r2_l11", action="store_true",
                        help="Build game_lora_r2_l11 DB (lora_r=2, q16, mlp=512, l11) before benchmarking")
    parser.add_argument("--game-lora-r2-l11-ckpt", dest="game_lora_r2_l11_ckpt", default=None,
                        help="Path to weights_end_lora2_l11_q16.pth checkpoint")
    parser.add_argument("--build-boq-l9", dest="build_boq_l9", action="store_true",
                        help="Build BoQ l9 descriptor DB from the l9 TRT engine before benchmarking")
    parser.add_argument("--build-vlad-l10", dest="build_vlad_l10", action="store_true",
                        help="Build layer-10 VLAD codebook + DB before benchmarking")
    parser.add_argument("--build-vlad-l11", dest="build_vlad_l11", action="store_true",
                        help="Build layer-11 VLAD codebook + DB before benchmarking")
    args = parser.parse_args()
    if args.bags:
        run_all_bags(args)
    else:
        run_bench(args)


if __name__ == "__main__":
    main()
