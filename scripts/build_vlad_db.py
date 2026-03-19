#!/usr/bin/env python3
"""
build_vlad_db.py — Build VLAD codebook + satellite patch descriptor database.

Steps:
  1. Load all satellite patches from data/patches/
  2. Extract DINOv3 layer-10 value features for each patch (batched)
  3. Subsample up to 100k patch-level feature vectors, run K-means (K=16)
  4. Save codebook  →  data/descriptors_vlad/vlad_codebook.pt
  5. Compute VLAD descriptor for each patch
  6. Save descriptors → data/descriptors_vlad/vlad_descriptors.pt
           names     → data/descriptors_vlad/patch_names.txt

Run from the repo root:
    python3 particle_filter_loc/scripts/build_vlad_db.py
or with a custom config:
    python3 particle_filter_loc/scripts/build_vlad_db.py --config <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# ── resolve project paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PF_ROOT = SCRIPT_DIR.parent                   # particle_filter_loc/
REPO_ROOT = PF_ROOT.parent                    # vpr_dinov3/

# Allow importing vlad_extractor from the package
sys.path.insert(0, str(PF_ROOT))
from particle_filter_loc.vlad_extractor import DINOv3VLADExtractor


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("matchers", cfg)


def _find_patches(patches_dir: Path) -> list[Path]:
    patches = sorted(patches_dir.glob("*.png"))
    if not patches:
        raise FileNotFoundError(f"No .png files found in {patches_dir}")
    return patches


def _extract_all_features(
    extractor: DINOv3VLADExtractor,
    patch_paths: list[Path],
    batch_size: int = 8,
) -> tuple[torch.Tensor, list[str]]:
    """
    Extract patch-level features for every patch image.

    Returns:
        all_feats:   [N_patches, N_tokens, D]  (float32, CPU)
        patch_names: list of stem names (without .png)
    """
    import cv2

    all_feats: list[torch.Tensor] = []
    patch_names: list[str] = []

    for i in tqdm(range(0, len(patch_paths), batch_size), desc="Extracting features"):
        batch_paths = patch_paths[i : i + batch_size]
        tensors = []
        names = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"  [warn] Could not read {p}, skipping")
                continue
            tensors.append(extractor.preprocess(img))
            names.append(p.stem)

        if not tensors:
            continue

        batch_tensor = torch.cat(tensors, dim=0)   # [B, 3, H, W]
        feats = extractor.extract_patch_features(batch_tensor)  # [B, N, D]
        all_feats.append(feats.cpu())
        patch_names.extend(names)

    return torch.cat(all_feats, dim=0), patch_names   # [N_patches, N, D]


def _fit_kmeans(
    all_feats: torch.Tensor,
    num_clusters: int,
    max_samples: int = 100_000,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Run K-means on a subsample of patch-level descriptors.

    Args:
        all_feats:    [N_patches, N_tokens, D]
        num_clusters: K
        max_samples:  cap on total token count fed to K-means
        device:       where to run K-means

    Returns:
        c_centers: [K, D] cluster centroids (float32, CPU)
    """
    try:
        import fast_pytorch_kmeans as fpk
    except ImportError:
        raise ImportError(
            "fast_pytorch_kmeans is required for codebook building. "
            "Install with:  pip install fast-pytorch-kmeans"
        )

    N_patches, N_tokens, D = all_feats.shape
    # Flatten to [N_patches * N_tokens, D]
    flat = all_feats.reshape(-1, D)

    # Subsample
    total = flat.shape[0]
    if total > max_samples:
        idx = torch.randperm(total)[:max_samples]
        flat = flat[idx]
    print(f"K-means: {flat.shape[0]} descriptors, K={num_clusters}, D={D}")

    flat = flat.to(device)
    kmeans = fpk.KMeans(n_clusters=num_clusters, mode="euclidean", verbose=1)
    kmeans.fit(flat)

    return kmeans.centroids.cpu().float()   # [K, D]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build VLAD descriptor database")
    parser.add_argument("--config", default=str(PF_ROOT / "config" / "pf_config.yaml"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_kmeans_samples", type=int, default=100_000)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))

    script_dir = Path(cfg.get("script_dir", str(REPO_ROOT / "dinov2_vits_layer_test")))

    # Paths from config (VLAD-specific keys)
    patches_dir    = script_dir / cfg.get("patches_dir",  "data/patches")
    codebook_path  = script_dir / cfg.get("vlad_codebook_path",
                                          "data/descriptors_vlad/vlad_codebook.pt")
    db_path        = script_dir / cfg.get("vlad_database_path",
                                          "data/descriptors_vlad/vlad_descriptors.pt")
    names_path     = script_dir / cfg.get("vlad_patch_names_path",
                                          "data/descriptors_vlad/patch_names.txt")
    num_clusters   = int(cfg.get("vlad_num_clusters", 16))
    layer_idx      = int(cfg.get("vlad_layer_idx",    10))
    image_size     = int(cfg.get("vlad_image_size",   256))
    grayscale      = bool(cfg.get("grayscale",         True))

    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    names_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Patches dir   : {patches_dir}")
    print(f"Codebook out  : {codebook_path}")
    print(f"Database out  : {db_path}")
    print(f"Names out     : {names_path}")
    print(f"K={num_clusters}, layer={layer_idx}, img_size={image_size}, grayscale={grayscale}")
    print()

    # ── 1. Load patch paths ──────────────────────────────────────────────
    patch_paths = _find_patches(patches_dir)
    print(f"Found {len(patch_paths)} patches")

    # ── 2. Feature extraction (no codebook needed yet) ──────────────────
    extractor = DINOv3VLADExtractor(
        codebook_path=None,
        layer_idx=layer_idx,
        image_size=image_size,
        grayscale=grayscale,
        device="cuda",
    )

    all_feats, patch_names = _extract_all_features(
        extractor, patch_paths, batch_size=args.batch_size
    )
    print(f"Features shape: {all_feats.shape}")   # [N, N_tokens, D]

    # ── 3. K-means codebook ──────────────────────────────────────────────
    c_centers = _fit_kmeans(
        all_feats,
        num_clusters=num_clusters,
        max_samples=args.max_kmeans_samples,
        device="cuda",
    )
    torch.save({"c_centers": c_centers}, str(codebook_path))
    print(f"Codebook saved → {codebook_path}  (shape: {c_centers.shape})")

    # ── 4. Compute VLAD for every patch ──────────────────────────────────
    extractor._load_codebook(str(codebook_path))

    all_vlad: list[torch.Tensor] = []
    for i in tqdm(range(len(patch_names)), desc="Computing VLAD"):
        feats_i = all_feats[i].unsqueeze(0).to(extractor.device)  # [1, N, D]
        vlad_i  = extractor._vlad_aggregate(feats_i).cpu()        # [1, K*D]
        all_vlad.append(vlad_i)

    vlad_db = torch.cat(all_vlad, dim=0)   # [N_patches, K*D]
    print(f"VLAD descriptors shape: {vlad_db.shape}")

    torch.save(vlad_db, str(db_path))
    print(f"Database saved → {db_path}")

    names_path.write_text("\n".join(patch_names) + "\n")
    print(f"Names saved    → {names_path}  ({len(patch_names)} entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
