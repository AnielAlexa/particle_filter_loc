"""DINOv3 ViT-S/16 + VLAD extractor for coarse place recognition.

Extracts patch-level value features from an intermediate layer of a
timm DINOv3 ViT-S/16 model and aggregates them with hard-assignment
VLAD against a pre-built codebook.

Output descriptor: [B, K * 384] L2-normalized   (K=16 → 6144-dim)
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm


class DINOv3VLADExtractor:
    """
    Coarse descriptor extractor: timm DINOv3 ViT-S/16 + VLAD.

    Args:
        codebook_path: Path to .pt file containing {'c_centers': [K, D]}.
                       If None, VLAD aggregation is disabled (feature
                       extraction only — used during codebook building).
        layer_idx:     Block index (0-indexed) to hook. Default 10.
        image_size:    Input spatial resolution (square). Default 256.
        grayscale:     If True, convert query frames to grayscale then
                       replicate to 3 channels before encoding.
        device:        Torch device string. Default 'cuda'.
    """

    MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"

    def __init__(
        self,
        codebook_path: Optional[str] = None,
        layer_idx: int = 10,
        image_size: int = 256,
        grayscale: bool = True,
        device: str = "cuda",
    ) -> None:
        self.layer_idx = layer_idx
        self.image_size = image_size
        self.grayscale = grayscale
        self.device = torch.device(device)
        self._hook_out: Optional[torch.Tensor] = None

        # ── backbone ────────────────────────────────────────────────────
        print(f"Loading {self.MODEL_NAME} (pretrained=True, img_size={image_size}) …")
        self.model = timm.create_model(
            self.MODEL_NAME,
            pretrained=True,
            num_classes=0,
            img_size=image_size,
        )
        self.model.eval().to(self.device)

        # Number of prefix tokens (CLS + register tokens)
        self.num_prefix_tokens: int = self.model.num_prefix_tokens

        # Normalization from timm model config (mean/std)
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.mean = torch.tensor(data_cfg["mean"], dtype=torch.float32,
                                 device=self.device).view(1, 3, 1, 1)
        self.std  = torch.tensor(data_cfg["std"],  dtype=torch.float32,
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

    # ── internal helpers ──────────────────────────────────────────────

    def _qkv_hook(self, module, inputs, output: torch.Tensor) -> None:
        """Captures the value facet from the QKV projection output."""
        B, N, _ = output.shape
        D = output.shape[-1] // 3
        # value is the last third: indices [2*D : 3*D]
        self._hook_out = output.reshape(B, N, 3, D)[:, :, 2, :]  # [B, N, D]

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

    # ── public API ────────────────────────────────────────────────────

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Center-crop to square, resize to image_size, optionally convert to
        grayscale (replicated to 3 ch), normalize with timm mean/std.

        Args:
            frame_bgr: H×W×3 uint8 BGR image (OpenCV format).

        Returns:
            Tensor [1, 3, image_size, image_size] on self.device.
        """
        h, w = frame_bgr.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        frame = frame_bgr[y0:y0 + s, x0:x0 + s]
        frame = cv2.resize(frame, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_CUBIC)

        if self.grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.stack([gray, gray, gray], axis=2)  # HWC, still uint8
        else:
            frame = frame[:, :, ::-1]  # BGR → RGB

        # HWC uint8 → NCHW float32 [0, 1]
        tensor = (
            torch.from_numpy(frame.transpose(2, 0, 1).copy())
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 255.0
        )
        tensor = (tensor - self.mean) / self.std
        return tensor

    def extract_patch_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and return L2-normalized patch features.

        Args:
            tensor: preprocessed image [B, 3, H, W].

        Returns:
            Patch features [B, N_patches, D] (prefix tokens stripped, L2-normed).
        """
        with torch.no_grad():
            self._hook_out = None
            self.model(tensor)
            assert self._hook_out is not None, "QKV hook did not fire"
            feats = self._hook_out[:, self.num_prefix_tokens:, :]  # strip prefix
            feats = F.normalize(feats, dim=-1)
        return feats

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline: backbone → patch features → VLAD aggregation.

        Args:
            tensor: preprocessed image [B, 3, H, W].

        Returns:
            L2-normalized VLAD descriptor [B, K*D].
        """
        feats = self.extract_patch_features(tensor)
        return self._vlad_aggregate(feats)

    def _vlad_aggregate(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Hard-assignment VLAD with intra-cluster normalization.

        Args:
            feats: L2-normalized patch features [B, N, D].

        Returns:
            [B, K*D] L2-normalized VLAD descriptor.
        """
        assert self.c_centers is not None, \
            "Codebook not loaded — call _load_codebook() or pass codebook_path in __init__"

        B, N, D = feats.shape
        K = self.K
        c = self.c_centers  # [K, D]

        # Squared Euclidean distances: ||f - c||^2 = 1 + ||c||^2 - 2 f·c
        # (feats are L2-normalized so ||f||^2 = 1)
        c_sq = (c ** 2).sum(-1)                             # [K]
        cross = torch.einsum("bnd,kd->bnk", feats, c)      # [B, N, K]
        dists = 1.0 + c_sq - 2.0 * cross                   # [B, N, K]
        assignments = dists.argmin(dim=-1)                  # [B, N]

        vlad = feats.new_zeros(B, K, D)
        for k in range(K):
            mask = (assignments == k).unsqueeze(-1).float()          # [B, N, 1]
            residuals = feats - c[k].unsqueeze(0).unsqueeze(0)       # [B, N, D]
            cluster_sum = (residuals * mask).sum(dim=1)              # [B, D]
            vlad[:, k] = F.normalize(cluster_sum, dim=-1)

        vlad = vlad.flatten(1)           # [B, K*D]
        return F.normalize(vlad, dim=-1)

    def __del__(self) -> None:
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
