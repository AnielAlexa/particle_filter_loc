#!/usr/bin/env python3
"""build_vlad_pca.py — Fit PCA on aerial (drone+satellite) VLAD descriptors and save.

Uses the same aerialvl100m JSON dataset as the aerial codebook builder.
Saves pca_mean.pt and pca_components.pt into the VLAD DB directory
so they can be loaded at runtime instead of computing PCA from the
(small, satellite-only) patch database.

Usage:
  python3 build_vlad_pca.py --pca-dim 768
  python3 build_vlad_pca.py --pca-dim 768 --max-images 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_DIR     = SCRIPTS_DIR.parent

sys.path.insert(0, str(PKG_DIR))
from particle_filter_loc.vlad_extractor import DINOv3VLADExtractor

DATA_ROOT   = Path("/media/aniel/storage/dataset_to_analize")
AERIAL_JSON = DATA_ROOT / "data" / "aerialvl100m_multi_vpair_train.json"


def collect_image_paths(json_path: Path, max_images: int, seed: int = 42) -> list[Path]:
    with open(str(json_path)) as f:
        entries = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(entries)
    paths: list[Path] = []
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
    print(f"Collected {len(paths)} image paths (cap={max_images})")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PCA on aerial VLAD descriptors")
    parser.add_argument("--config", default=str(PKG_DIR / "config" / "pf_config.yaml"))
    parser.add_argument("--pca-dim", dest="pca_dim", type=int, default=768)
    parser.add_argument("--max-images", dest="max_images", type=int, default=10_000)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["matchers"]

    script_dir = Path(mcfg["script_dir"])
    db_dir = script_dir / Path(mcfg["vlad_database_path"]).parent
    cb_path = script_dir / mcfg["vlad_codebook_path"]
    db_path = script_dir / mcfg["vlad_database_path"]
    layer_idx = int(mcfg.get("vlad_layer_idx", 10))
    image_size = int(mcfg.get("vlad_image_size", 256))
    grayscale = bool(mcfg.get("grayscale", True))

    print(f"DB dir:     {db_dir}")
    print(f"Codebook:   {cb_path}")
    print(f"PCA dim:    {args.pca_dim}")
    print(f"Layer:      {layer_idx}")

    # 1. Collect aerial image paths
    image_paths = collect_image_paths(AERIAL_JSON, args.max_images)

    # 2. Load extractor with aerial codebook
    extractor = DINOv3VLADExtractor(
        codebook_path=str(cb_path),
        layer_idx=layer_idx,
        image_size=image_size,
        grayscale=grayscale,
        device="cuda",
    )

    # 3. Compute VLAD descriptors for aerial images
    print(f"\nExtracting VLAD descriptors from {len(image_paths)} aerial images ...")
    all_vlad: list[torch.Tensor] = []
    failed = 0

    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="VLAD extraction"):
        batch_paths = image_paths[i : i + args.batch_size]
        tensors = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                failed += 1
                continue
            tensors.append(extractor.preprocess(img))
        if not tensors:
            continue
        batch_t = torch.cat(tensors, dim=0)
        feats = extractor.extract_patch_features(batch_t)  # [B, N, D]
        for j in range(feats.shape[0]):
            v = extractor._vlad_aggregate(feats[j].unsqueeze(0).to(extractor.device))
            all_vlad.append(v.cpu())

    if failed:
        print(f"  [warn] {failed} images could not be read")

    aerial_vlad = torch.cat(all_vlad, dim=0)  # [N_images, K*D]
    print(f"Aerial VLAD: {aerial_vlad.shape}")

    # 4. Fit PCA on aerial descriptors
    print(f"\nFitting PCA: {aerial_vlad.shape[1]} → {args.pca_dim} ...")
    aerial_vlad = F.normalize(aerial_vlad.float().cuda(), dim=1)
    mean = aerial_vlad.mean(dim=0)
    X_c = aerial_vlad - mean
    _, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
    k = min(args.pca_dim, Vh.shape[0])
    explained = (S[:k] ** 2).sum() / (S ** 2).sum()
    components = Vh[:k]  # [k, D]
    print(f"PCA: {aerial_vlad.shape[1]}→{k} dims, explained variance: {explained:.1%}")

    # 5. Save PCA components
    pca_path = db_dir / "pca_components.pt"
    torch.save({
        "mean": mean.cpu(),
        "components": components.cpu(),
        "pca_dim": k,
        "source_dim": aerial_vlad.shape[1],
        "n_train_samples": aerial_vlad.shape[0],
        "explained_variance": float(explained),
    }, str(pca_path))
    print(f"PCA saved: {pca_path}")

    # 6. Also project and save PCA-reduced patch DB
    print(f"\nProjecting patch DB through PCA ...")
    db_data = torch.load(str(db_path), map_location="cuda", weights_only=False)
    if isinstance(db_data, dict):
        db = db_data.get("descriptors", list(db_data.values())[0])
    else:
        db = db_data
    db = F.normalize(db.float().cuda(), dim=1)
    db_pca = F.normalize((db - mean) @ components.T, dim=1)
    pca_db_path = db_dir / f"vlad_descriptors_pca{k}.pt"
    torch.save(db_pca.cpu(), str(pca_db_path))
    print(f"PCA DB saved: {pca_db_path}  shape={db_pca.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
