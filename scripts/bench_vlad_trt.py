#!/usr/bin/env python3
"""bench_vlad_trt.py

Benchmark PyTorch vs TensorRT VLAD+PCA768 coarse retrieval on an MCAP bag.

Metrics
-------
  Speed         : Inference time (mean, median, p95) per method
  Fidelity      : Cosine similarity between PyTorch and TRT descriptors
  Retrieval     : Top-1 agreement, Recall@1, Recall@5, localization error
  Rank corr.    : Spearman rank correlation of top-K retrieval lists
  Engine size   : File size comparison

Usage
-----
  python bench_vlad_trt.py
  python bench_vlad_trt.py --bag /path/to/Day2.6_0.mcap --start-offset 55
  python bench_vlad_trt.py --max-frames 200
  python bench_vlad_trt.py --use-trt-db   # also bench with TRT-generated database
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ── project paths ─────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_DIR     = SCRIPTS_DIR.parent
REPO_ROOT   = PKG_DIR.parent

sys.path.insert(0, str(PKG_DIR))
from particle_filter_loc.geo_utils import ENUFrame, haversine_m
from particle_filter_loc.vlad_extractor import DINOv3VLADExtractor


# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BAG    = "/media/aniel/storage/bags_jetson/095c6d2c7237fe986c6240628b6ef8dd/Day2.6_0.mcap"
DEFAULT_CONFIG = str(PKG_DIR / "config" / "pf_config.yaml")
DEFAULT_START  = 55.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with open(path) as f:
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
        if m:
            e, n = enu.wgs84_to_enu(m["lat"], m["lon"])
        else:
            e, n = 0.0, 0.0
        coords.append((e, n))
    return np.array(coords, dtype=np.float64)


# ── Extractor wrappers ────────────────────────────────────────────────────────

class VLADPCAExtractor:
    """PyTorch DINOv3VLADExtractor + PCA projection."""

    def __init__(self, codebook_path, pca_path, layer_idx=10,
                 image_size=256, grayscale=True, pca_dim=768, device="cuda"):
        self.base = DINOv3VLADExtractor(
            codebook_path=codebook_path,
            layer_idx=layer_idx,
            image_size=image_size,
            grayscale=grayscale,
            device=device,
        )
        pca_data = torch.load(pca_path, map_location=device, weights_only=False)
        self._pca_mean = pca_data["mean"].to(device)
        self._pca_components = pca_data["components"][:pca_dim].to(device)
        self.device = torch.device(device)

    def preprocess(self, frame_bgr):
        return self.base.preprocess(frame_bgr)

    def __call__(self, tensor):
        vlad = self.base(tensor).float().to(self.device)
        projected = (vlad - self._pca_mean) @ self._pca_components.T
        return F.normalize(projected, dim=1)


@dataclass
class MethodResult:
    descriptor: torch.Tensor   # [1, D]
    top1_idx: int
    top1_sim: float
    top5_idx: np.ndarray
    time_ms: float


@dataclass
class Method:
    name: str
    extractor: object
    db: torch.Tensor           # [N, D] cuda
    patch_names: List[str]
    patch_enu: np.ndarray      # [N, 2]

    def run(self, frame_bgr: np.ndarray) -> MethodResult:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tensor = self.extractor.preprocess(frame_bgr)
        with torch.no_grad():
            desc = F.normalize(self.extractor(tensor).float(), dim=1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        sims = (desc @ self.db.T).squeeze(0)
        top1_idx = int(sims.argmax().item())
        top5_idx = sims.topk(5).indices.cpu().numpy()
        return MethodResult(
            descriptor=desc,
            top1_idx=top1_idx,
            top1_sim=float(sims[top1_idx].item()),
            top5_idx=top5_idx,
            time_ms=(t1 - t0) * 1000,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_bench(args):
    cfg = _load_config(Path(args.config))
    mcfg = cfg["matchers"]
    rcfg = cfg["replay"]

    script_dir = Path(mcfg["script_dir"])
    image_size = int(mcfg.get("image_size", 256))
    grayscale  = bool(mcfg.get("grayscale", True))

    vlad_dir = script_dir / "data" / "descriptors_vlad_l10_aerial"
    codebook_path = str(vlad_dir / "vlad_codebook.pt")
    pca_path      = str(vlad_dir / "pca_components.pt")
    vlad_db_path  = vlad_dir / "vlad_descriptors.pt"
    names_path    = vlad_dir / "patch_names.txt"

    # GPS metadata
    meta_path = script_dir / mcfg.get("gps_metadata_path", "data/patches/gps_metadata.json")
    with open(str(meta_path)) as f:
        gps_meta = json.load(f)
    enu_orig = cfg["enu_origin"]
    enu = ENUFrame(enu_orig["lat"], enu_orig["lon"])

    methods: List[Method] = []

    # ── Method 1: PyTorch VLAD+PCA768 ─────────────────────────────────────
    print("[vlad_pytorch] Loading PyTorch VLAD+PCA extractor...")
    pt_ext = VLADPCAExtractor(
        codebook_path=codebook_path, pca_path=pca_path,
        layer_idx=10, image_size=image_size, grayscale=grayscale,
    )
    # Load full VLAD DB and project through PCA for fair comparison
    raw_db = torch.load(str(vlad_db_path), map_location="cuda", weights_only=False)
    if isinstance(raw_db, dict):
        raw_db = raw_db.get("descriptors", list(raw_db.values())[0])
    raw_db = raw_db.float().cuda()
    pca_data = torch.load(pca_path, map_location="cuda", weights_only=False)
    pca_mean = pca_data["mean"].cuda()
    pca_comp = pca_data["components"][:768].cuda()
    db_pca = F.normalize((raw_db - pca_mean) @ pca_comp.T, dim=1)
    with open(str(names_path)) as f:
        pt_names = [l.strip() for l in f if l.strip()]
    pt_enu = _patch_enu_array(pt_names, gps_meta, enu)
    methods.append(Method("vlad_pytorch", pt_ext, db_pca, pt_names, pt_enu))
    print(f"  DB: {db_pca.shape}")

    # ── Method 2: TensorRT VLAD+PCA768 ────────────────────────────────────
    trt_engine = script_dir / "dinov3_vlad_pca768_l10_value_256.engine"
    if not trt_engine.exists():
        print(f"[warn] TRT engine not found: {trt_engine}")
        print("  Run conversion/convert_dinov3_vlad_pca_trt.py first.")
    else:
        print(f"[vlad_trt] Loading TRT engine: {trt_engine}")
        mod = _import_match_module(str(script_dir))
        trt_ext = mod.DINOv3VLADPCAExtractorTRT(
            engine_path=str(trt_engine),
            image_size=image_size,
            grayscale=grayscale,
        )
        # Use same PCA-projected database for fair comparison
        methods.append(Method("vlad_trt", trt_ext, db_pca.clone(), pt_names, pt_enu))
        print(f"  DB: {db_pca.shape} (shared with PyTorch)")

    # ── Optional: TRT with TRT-generated database ─────────────────────────
    if args.use_trt_db:
        trt_db_dir = script_dir / "data" / "descriptors_vlad_l10_aerial_trt"
        trt_db_path = trt_db_dir / "vlad_pca_trt_descriptors.pt"
        trt_names_path = trt_db_dir / "patch_names.txt"
        if trt_db_path.exists() and trt_engine.exists():
            print("[vlad_trt_owndb] Loading TRT engine + TRT-generated DB...")
            trt_ext2 = mod.DINOv3VLADPCAExtractorTRT(
                engine_path=str(trt_engine),
                image_size=image_size,
                grayscale=grayscale,
            )
            db_trt, trt_names = _load_db(trt_db_path, trt_names_path)
            trt_enu = _patch_enu_array(trt_names, gps_meta, enu)
            methods.append(Method("vlad_trt_owndb", trt_ext2, db_trt, trt_names, trt_enu))
            print(f"  DB: {db_trt.shape}")
        else:
            print("[warn] TRT DB not found. Run generate_vlad_pca_database.py first.")

    if len(methods) < 2:
        print("[error] Need at least 2 methods (PyTorch + TRT). Aborting.")
        return

    # ── Warmup ────────────────────────────────────────────────────────────
    print("\nWarming up extractors...")
    dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for m in methods:
        for _ in range(3):
            m.run(dummy)
    print("  Warmup done.")

    # ── MCAP setup ────────────────────────────────────────────────────────
    mcap_path  = args.bag or DEFAULT_BAG
    start_off  = args.start_offset if args.start_offset is not None else DEFAULT_START
    cam_topic  = rcfg.get("camera_topic", "/camera/image_mono")
    rtk_topic  = rcfg.get("rtk_topic", "/m300/rtk/fix")
    alt_topic  = rcfg.get("altimeter_topic", "/altimeter/range")
    alt_min    = rcfg.get("altitude_min_process_m", 0.0)
    subsample  = args.subsample or rcfg.get("camera_subsample", 1)
    gt_radius  = args.gt_radius

    bag_name = args.bag_name or Path(mcap_path).stem.replace("_0", "")
    out_dir  = PKG_DIR / "results" / "bench_vlad_trt" / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[mcap] Bag: {mcap_path}")
    print(f"[mcap] start_offset={start_off}s  alt_min={alt_min}m  gt_radius={gt_radius}m")

    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    # ── Accumulators ──────────────────────────────────────────────────────
    timings: Dict[str, List[float]] = {m.name: [] for m in methods}
    fidelity_sims: List[float] = []
    top1_agree: List[int] = []
    per_method_metrics: Dict[str, List[dict]] = {m.name: [] for m in methods}
    n_frames = 0

    gt_lat, gt_lon = None, None
    altitude_buf   = deque(maxlen=10)
    current_alt    = 0.0
    t_start        = None
    cam_idx        = 0

    with open(str(mcap_path), "rb") as bag_f:
        reader = make_reader(bag_f, decoder_factories=[DecoderFactory()])
        topics = [cam_topic, rtk_topic, alt_topic]

        for schema, channel, message, decoded_msg in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            ts_ns = message.log_time

            if t_start is None:
                t_start = ts_ns
            elapsed = (ts_ns - t_start) * 1e-9

            if elapsed < start_off:
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

            cam_idx += 1
            if cam_idx % subsample != 0:
                continue
            if args.max_frames and cam_idx > args.max_frames:
                print(f"[mcap] Reached max_frames={args.max_frames}")
                break
            if alt_min > 0 and current_alt < alt_min:
                continue

            # Decode image
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

            if gt_lat is None:
                continue

            rtk_e, rtk_n = enu.wgs84_to_enu(gt_lat, gt_lon)

            # GT patch
            ref = methods[0]
            dx = ref.patch_enu[:, 0] - rtk_e
            dy = ref.patch_enu[:, 1] - rtk_n
            dists = np.hypot(dx, dy)
            gt_idx_ref = int(dists.argmin())
            has_gt = dists[gt_idx_ref] <= gt_radius

            # Run all methods
            results: Dict[str, MethodResult] = {}
            for m in methods:
                results[m.name] = m.run(frame_bgr)
                timings[m.name].append(results[m.name].time_ms)

            # Fidelity: cosine sim between PyTorch and TRT descriptors
            if "vlad_pytorch" in results and "vlad_trt" in results:
                sim = F.cosine_similarity(
                    results["vlad_pytorch"].descriptor,
                    results["vlad_trt"].descriptor,
                ).item()
                fidelity_sims.append(sim)

            # Top-1 agreement
            if "vlad_pytorch" in results and "vlad_trt" in results:
                agree = int(results["vlad_pytorch"].top1_idx == results["vlad_trt"].top1_idx)
                top1_agree.append(agree)

            # Per-method metrics
            for m in methods:
                r = results[m.name]
                top1_e, top1_n = m.patch_enu[r.top1_idx]
                error_m = math.hypot(top1_e - rtk_e, top1_n - rtk_n)

                if has_gt:
                    gt_name = ref.patch_names[gt_idx_ref]
                    try:
                        gt_idx = m.patch_names.index(gt_name)
                    except ValueError:
                        gt_idx = -1
                    sims_all = (r.descriptor @ m.db.T).squeeze(0)
                    r1 = int(r.top1_idx == gt_idx)
                    r5 = int(gt_idx in r.top5_idx)
                else:
                    r1 = r5 = -1

                per_method_metrics[m.name].append({
                    "error_m": error_m, "recall1": r1, "recall5": r5,
                })

            n_frames += 1
            if n_frames % 50 == 1:
                err_str = "  ".join(
                    f"{m.name}:{per_method_metrics[m.name][-1]['error_m']:.0f}m"
                    for m in methods
                )
                fidel = f"  fid={fidelity_sims[-1]:.4f}" if fidelity_sims else ""
                print(f"  frame {cam_idx:4d}  t={elapsed:.1f}s  alt={current_alt:.0f}m  "
                      f"{err_str}{fidel}")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"Benchmark Results — {bag_name} — {n_frames} frames")
    print(f"{'=' * 70}")

    # Speed
    print(f"\n--- Speed (ms/frame) ---")
    for m in methods:
        t = np.array(timings[m.name])
        print(f"  {m.name:25s}  mean={t.mean():.1f}  median={np.median(t):.1f}  "
              f"p95={np.percentile(t, 95):.1f}  min={t.min():.1f}  max={t.max():.1f}")

    # Fidelity
    if fidelity_sims:
        fs = np.array(fidelity_sims)
        print(f"\n--- Fidelity (PT vs TRT descriptor cosine similarity) ---")
        print(f"  mean={fs.mean():.6f}  median={np.median(fs):.6f}  "
              f"min={fs.min():.6f}  p5={np.percentile(fs, 5):.6f}  "
              f"p95={np.percentile(fs, 95):.6f}")

    # Agreement
    if top1_agree:
        pct = np.mean(top1_agree) * 100
        print(f"\n--- Top-1 Retrieval Agreement ---")
        print(f"  {pct:.1f}% ({sum(top1_agree)}/{len(top1_agree)} frames)")

    # Recall & error
    print(f"\n--- Retrieval Quality ---")
    for m in methods:
        rows = per_method_metrics[m.name]
        valid = [r for r in rows if r["recall1"] >= 0]
        if not valid:
            print(f"  {m.name}: no GT frames")
            continue
        errors = np.array([r["error_m"] for r in valid])
        r1 = np.mean([r["recall1"] for r in valid]) * 100
        r5 = np.mean([r["recall5"] for r in valid]) * 100
        print(f"  {m.name:25s}  R@1={r1:.1f}%  R@5={r5:.1f}%  "
              f"med_err={np.median(errors):.1f}m  mean_err={errors.mean():.1f}m  "
              f"p90_err={np.percentile(errors, 90):.1f}m")

    # Rank correlation
    if "vlad_pytorch" in results and "vlad_trt" in results:
        try:
            from scipy.stats import spearmanr
            # Compute on last frame's full similarity vectors
            pt_sims = (results["vlad_pytorch"].descriptor @ methods[0].db.T).squeeze(0).cpu().numpy()
            trt_sims = (results["vlad_trt"].descriptor @ methods[1].db.T).squeeze(0).cpu().numpy()
            rho, pval = spearmanr(pt_sims, trt_sims)
            print(f"\n--- Spearman Rank Correlation (last frame) ---")
            print(f"  rho={rho:.6f}  p={pval:.2e}")
        except ImportError:
            pass

    # Engine sizes
    print(f"\n--- File Sizes ---")
    trt_path = script_dir / "dinov3_vlad_pca768_l10_value_256.engine"
    onnx_path = script_dir / "dinov3_vlad_pca768_l10_value_256.onnx"
    for p, label in [(trt_path, "TRT engine"), (onnx_path, "ONNX model")]:
        if p.exists():
            print(f"  {label}: {p.stat().st_size / 1024 / 1024:.1f} MB")

    # Save summary CSV
    summary_path = out_dir / "summary.csv"
    with open(str(summary_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_frames", "speed_mean_ms", "speed_median_ms", "speed_p95_ms",
                     "recall1_pct", "recall5_pct", "median_error_m", "p90_error_m",
                     "fidelity_mean", "fidelity_min", "top1_agree_pct"])
        for m in methods:
            t = np.array(timings[m.name])
            rows = per_method_metrics[m.name]
            valid = [r for r in rows if r["recall1"] >= 0]
            errors = np.array([r["error_m"] for r in valid]) if valid else np.array([])
            r1 = np.mean([r["recall1"] for r in valid]) * 100 if valid else float("nan")
            r5 = np.mean([r["recall5"] for r in valid]) * 100 if valid else float("nan")

            fid_mean = np.mean(fidelity_sims) if fidelity_sims and m.name == "vlad_trt" else ""
            fid_min  = np.min(fidelity_sims)  if fidelity_sims and m.name == "vlad_trt" else ""
            agree    = np.mean(top1_agree)*100 if top1_agree and m.name == "vlad_trt" else ""

            w.writerow([
                m.name, n_frames,
                f"{t.mean():.1f}", f"{np.median(t):.1f}", f"{np.percentile(t, 95):.1f}",
                f"{r1:.1f}", f"{r5:.1f}",
                f"{np.median(errors):.1f}" if len(errors) else "",
                f"{np.percentile(errors, 90):.1f}" if len(errors) else "",
                f"{fid_mean:.6f}" if fid_mean != "" else "",
                f"{fid_min:.6f}" if fid_min != "" else "",
                f"{agree:.1f}" if agree != "" else "",
            ])

    print(f"\nSummary saved: {summary_path}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TRT VLAD+PCA768")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--bag", type=str, default=None,
                        help="MCAP bag path (default: Day2.6)")
    parser.add_argument("--bag-name", type=str, default=None)
    parser.add_argument("--start-offset", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--gt-radius", type=float, default=50.0,
                        help="Max distance (m) for a patch to be considered GT")
    parser.add_argument("--use-trt-db", action="store_true",
                        help="Also benchmark with TRT-generated descriptor database")
    args = parser.parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
