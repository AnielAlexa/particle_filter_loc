"""Footprint-aware helpers for coarse/fine matching parameterization."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FootprintInfo:
    altitude_m: float
    drone_gsd: float          # m/px in drone image (h/fx)
    sat_gsd: float            # m/px in satellite tiles
    footprint_w_m: float      # ground width drone sees (h * W / fx)
    footprint_h_m: float      # ground height drone sees (h * H / fy)
    scale_ratio: float        # drone_gsd / sat_gsd (< 1 means drone higher res)
    n_patches_spanned: float  # max(footprint_w, footprint_h) / patch_ground_size


def compute_footprint(
    fx: float, fy: float,
    img_w: int, img_h: int,
    altitude_m: float,
    sat_gsd: float = 0.298,
    patch_ground_size_m: float = 100.0,
) -> FootprintInfo:
    drone_gsd = altitude_m / fx
    footprint_w = altitude_m * img_w / fx
    footprint_h = altitude_m * img_h / fy
    scale_ratio = drone_gsd / sat_gsd
    n_patches = max(footprint_w, footprint_h) / patch_ground_size_m
    return FootprintInfo(
        altitude_m=altitude_m,
        drone_gsd=drone_gsd,
        sat_gsd=sat_gsd,
        footprint_w_m=footprint_w,
        footprint_h_m=footprint_h,
        scale_ratio=scale_ratio,
        n_patches_spanned=n_patches,
    )


def sigma_obs_coarse(footprint: FootprintInfo, base_sigma: float = 50.0) -> float:
    """Scale coarse sigma with number of patches spanned; floor at half the footprint."""
    floor = max(footprint.footprint_w_m, footprint.footprint_h_m) / 2.0
    scaled = base_sigma * max(1.0, footprint.n_patches_spanned)
    return max(scaled, floor)


def sigma_obs_fine(footprint: FootprintInfo, base_sigma: float = 5.0) -> float:
    """Scale fine sigma when drone resolution is lower than satellite (scale_ratio > 1)."""
    return base_sigma * max(1.0, footprint.scale_ratio)


def context_fraction_from_footprint(
    footprint: FootprintInfo,
    patch_ground_size_m: float = 100.0,
    margin_factor: float = 1.3,
) -> float:
    """Compute context_fraction so composite covers drone FOV + margin.

    context_fraction is the extension beyond the center patch in each direction,
    as a fraction of one patch span.
    """
    needed_span = max(footprint.footprint_w_m, footprint.footprint_h_m) * margin_factor
    # The center patch covers patch_ground_size_m; we need (needed_span - patch) / 2
    # on each side, expressed as fraction of patch_ground_size_m
    extra_each_side = max(0.0, (needed_span - patch_ground_size_m) / 2.0)
    return extra_each_side / patch_ground_size_m


def scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float,
    orig_w: int, orig_h: int,
    target_w: int, target_h: int,
) -> Tuple[float, float, float, float]:
    """Scale camera intrinsics from original resolution to target resolution."""
    sx = target_w / orig_w
    sy = target_h / orig_h
    return fx * sx, fy * sy, cx * sx, cy * sy
