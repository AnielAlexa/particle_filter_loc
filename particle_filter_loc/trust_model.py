"""Unified trust mechanism for particle filter geo-localization.

Computes composite confidence scores from multiple signals using a weighted
geometric mean.  Any single catastrophically low signal pulls the total down,
providing safety against bad matches.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TrustConfig:
    # Signal 1: Inlier score
    inlier_tau: float = 25.0
    inlier_weight: float = 0.35
    # Signal 2: Similarity score
    sim_floor: float = 0.10
    sim_ceiling: float = 0.55
    sim_weight: float = 0.20
    # Signal 3: PF consistency
    consistency_weight: float = 0.15
    consistency_min_radius_m: float = 30.0
    # Signal 4: Cross-agreement
    agreement_weight: float = 0.10
    agreement_scale_m: float = 40.0
    # Signal 5: Altitude
    altitude_weight: float = 0.10
    altitude_ref_m: float = 60.0
    # Signal 6: Temporal consistency
    temporal_weight: float = 0.10
    temporal_max_speed_m_s: float = 15.0
    # Sigma modulation
    sigma_min_scale: float = 1.0
    sigma_max_scale: float = 5.0
    # Drift detection
    drift_pf_confidence_threshold: float = 0.3
    drift_recon_sim_threshold: float = 0.20
    drift_coarse_sim_min: float = 0.35
    drift_sigma_cap: float = 3.0
    # High-confidence joint gate: recon_sim high + many inliers → confidence boost
    # Applies only to satellite/mosaic candidates (recon-based paths).
    # Boost is a multiplier on the final confidence score, clamped to 1.0.
    recon_high_conf_sim_thr: float = 0.30
    recon_high_conf_inlier_thr: int = 25
    recon_high_conf_boost: float = 1.5
    # Global correction: relocalize from coarse+fine when reconstruction diverges
    global_corr_enabled: bool = True
    global_corr_recon_sim_threshold: float = 0.15
    global_corr_min_frames: int = 3
    global_corr_min_inliers: int = 10
    global_corr_max_spread_m: float = 50.0
    global_corr_position_consistency_m: float = 40.0
    global_corr_cooldown_frames: int = 10
    global_corr_teleport_fraction: float = 0.35
    global_corr_teleport_sigma: float = 10.0
    # EMA smoothing
    ema_alpha: float = 0.3
    # Minimum confidence to apply any update
    min_confidence: float = 0.05
    # Coarse teleport (continuous)
    coarse_sim_floor: float = 0.25
    coarse_sim_ceiling: float = 0.60
    coarse_max_teleport_fraction: float = 0.8
    coarse_teleport_sigma_range: List[float] = field(
        default_factory=lambda: [15.0, 30.0]
    )


@dataclass
class CandidateScore:
    """Per-candidate trust breakdown."""
    east_m: float
    north_m: float
    inliers: int
    heading_deg: Optional[float]
    source: str
    # Individual signal scores
    inlier_score: float = 0.0
    sim_score: float = 0.0
    consistency_score: float = 0.0
    agreement_score: float = 0.0
    altitude_score: float = 0.0
    temporal_score: float = 0.0
    # Combined
    confidence: float = 0.0


@dataclass
class FrameTrust:
    """Per-frame trust result."""
    best: Optional[CandidateScore] = None
    all_scores: List[CandidateScore] = field(default_factory=list)
    drift_detected: bool = False
    recon_sim_ema: float = 0.0
    pf_self_confidence_ema: float = 0.0


@dataclass
class GlobalCorrectionResult:
    """Result of a global correction triggered by reconstruction divergence."""
    east_m: float
    north_m: float
    heading_deg: Optional[float]
    inliers_median: int
    n_consistent_frames: int
    distance_from_pf_m: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def score_candidate(
    east_m: float,
    north_m: float,
    inliers: int,
    heading_deg: Optional[float],
    source: str,
    sim: float,
    pf_east: float,
    pf_north: float,
    pf_spread: float,
    lost_spread: float,
    other_candidates_enu: List[Tuple[float, float]],
    altitude_m: float,
    prev_best_enu: Optional[Tuple[float, float]],
    dt_s: float,
    cfg: TrustConfig,
) -> CandidateScore:
    cs = CandidateScore(
        east_m=east_m, north_m=north_m, inliers=inliers,
        heading_deg=heading_deg, source=source,
    )

    # Signal 1: Inlier score — saturating exponential
    cs.inlier_score = 1.0 - math.exp(-inliers / cfg.inlier_tau)

    # Signal 2: Similarity score — linear ramp with dead zone
    if cfg.sim_ceiling > cfg.sim_floor:
        cs.sim_score = _clamp(
            (sim - cfg.sim_floor) / (cfg.sim_ceiling - cfg.sim_floor), 0.0, 1.0
        )
    else:
        cs.sim_score = 1.0 if sim >= cfg.sim_floor else 0.0

    # Signal 3: PF consistency — Gaussian penalty for distance from estimate
    if pf_spread > lost_spread:
        cs.consistency_score = 1.0  # PF is lost, accept anything
    else:
        effective_radius = max(pf_spread * 2.0, cfg.consistency_min_radius_m)
        dist = math.sqrt((east_m - pf_east) ** 2 + (north_m - pf_north) ** 2)
        cs.consistency_score = math.exp(-0.5 * (dist / effective_radius) ** 2)

    # Signal 4: Cross-agreement — reward agreement with other candidates
    if len(other_candidates_enu) == 0:
        cs.agreement_score = 0.5  # neutral for single candidate
    else:
        min_dist = min(
            math.sqrt((east_m - oe) ** 2 + (north_m - on) ** 2)
            for oe, on in other_candidates_enu
        )
        cs.agreement_score = math.exp(-min_dist / cfg.agreement_scale_m)

    # Signal 5: Altitude — higher altitude = less precise
    cs.altitude_score = _clamp(
        cfg.altitude_ref_m / max(altitude_m, 1.0), 0.3, 1.0
    )

    # Signal 6: Temporal consistency — distance from previous best
    if prev_best_enu is None:
        cs.temporal_score = 0.5  # neutral
    else:
        dt_dist = math.sqrt(
            (east_m - prev_best_enu[0]) ** 2 + (north_m - prev_best_enu[1]) ** 2
        )
        max_move = cfg.temporal_max_speed_m_s * max(dt_s, 0.01)
        scale = max(max_move * 2.0, 10.0)
        cs.temporal_score = math.exp(-0.5 * (dt_dist / scale) ** 2)

    # Combined: weighted geometric mean
    signals = [
        (cs.inlier_score, cfg.inlier_weight),
        (cs.sim_score, cfg.sim_weight),
        (cs.consistency_score, cfg.consistency_weight),
        (cs.agreement_score, cfg.agreement_weight),
        (cs.altitude_score, cfg.altitude_weight),
        (cs.temporal_score, cfg.temporal_weight),
    ]
    log_conf = sum(w * math.log(max(s, 1e-10)) for s, w in signals)
    cs.confidence = math.exp(log_conf)

    # High-confidence joint gate: recon_sim (= sim for sat/mosaic) high AND
    # many inliers → multiplicative boost so get_effective_sigma() gives a
    # tighter Gaussian → stronger particle pull toward this position.
    if (source in ("satellite", "mosaic")
            and sim >= cfg.recon_high_conf_sim_thr
            and inliers >= cfg.recon_high_conf_inlier_thr):
        cs.confidence = min(1.0, cs.confidence * cfg.recon_high_conf_boost)

    return cs


class TrustTracker:
    """Maintains EMA state across frames for temporal signals."""

    def __init__(self, cfg: TrustConfig):
        self.cfg = cfg
        self.recon_sim_ema: float = 0.0
        self.pf_self_confidence_ema: float = 0.5
        self.prev_best_enu: Optional[Tuple[float, float]] = None
        self.prev_best_ts: Optional[float] = None
        self._initialized = False
        # Global correction state
        self._recon_diverge_count: int = 0
        self._coarse_fine_hits: List[Tuple[float, float, int, Optional[float]]] = []
        self._global_correction_cooldown: int = 0

    def _ema(self, old: float, new: float) -> float:
        a = self.cfg.ema_alpha
        if not self._initialized:
            return new
        return a * new + (1.0 - a) * old

    def evaluate_frame(
        self,
        fine_candidates: List[Tuple],
        recon_sim: float,
        top1_sim: float,
        pf_east: float,
        pf_north: float,
        pf_spread: float,
        lost_spread: float,
        altitude_m: float,
        timestamp_s: float,
    ) -> FrameTrust:
        """Evaluate all fine candidates for a frame.

        fine_candidates: list of (east_m, north_m, inliers, heading_deg, source) tuples
        """
        # Update EMAs
        self.recon_sim_ema = self._ema(self.recon_sim_ema, recon_sim)
        pf_self_conf = 1.0 - _clamp(pf_spread / lost_spread, 0.0, 1.0)
        self.pf_self_confidence_ema = self._ema(self.pf_self_confidence_ema, pf_self_conf)
        self._initialized = True

        # Drift detection
        drift_detected = (
            self.pf_self_confidence_ema < self.cfg.drift_pf_confidence_threshold
            and self.recon_sim_ema < self.cfg.drift_recon_sim_threshold
            and top1_sim >= self.cfg.drift_coarse_sim_min
        )

        # Compute dt from previous frame
        dt_s = 0.0
        if self.prev_best_ts is not None:
            dt_s = timestamp_s - self.prev_best_ts

        # Build list of all candidate positions for cross-agreement
        all_enu = [(e, n) for e, n, _, _, _ in fine_candidates]

        result = FrameTrust(
            drift_detected=drift_detected,
            recon_sim_ema=self.recon_sim_ema,
            pf_self_confidence_ema=self.pf_self_confidence_ema,
        )

        for e, n, inliers, hdg, source in fine_candidates:
            # When drift detected, drop mosaic/satellite candidates
            if drift_detected and source in ("satellite", "mosaic"):
                continue

            # Determine which similarity to use
            if source in ("satellite", "mosaic"):
                sim = recon_sim
            else:
                sim = top1_sim

            # Other candidates for cross-agreement (exclude self)
            others = [(oe, on) for oe, on in all_enu if (oe, on) != (e, n)]

            # When drift, force consistency to 1.0 for coarse candidates
            effective_spread = lost_spread + 1.0 if drift_detected else pf_spread

            cs = score_candidate(
                east_m=e, north_m=n, inliers=inliers,
                heading_deg=hdg, source=source, sim=sim,
                pf_east=pf_east, pf_north=pf_north,
                pf_spread=effective_spread, lost_spread=lost_spread,
                other_candidates_enu=others,
                altitude_m=altitude_m,
                prev_best_enu=self.prev_best_enu,
                dt_s=dt_s,
                cfg=self.cfg,
            )
            result.all_scores.append(cs)

        # Pick best candidate above minimum confidence
        if result.all_scores:
            best = max(result.all_scores, key=lambda s: s.confidence)
            if best.confidence >= self.cfg.min_confidence:
                result.best = best

        # Update temporal state
        if result.best is not None:
            self.prev_best_enu = (result.best.east_m, result.best.north_m)
            self.prev_best_ts = timestamp_s

        return result

    def get_sigma_scale(self, confidence: float) -> float:
        """Map confidence to sigma scale factor."""
        # confidence=0 -> sigma_max_scale, confidence=1 -> sigma_min_scale
        return 1.0 / (0.2 + 0.8 * confidence)

    def get_effective_sigma(self, confidence: float, base_sigma: float,
                            is_static: bool = False) -> float:
        scale = self.get_sigma_scale(confidence)
        # Cap sigma scale during drift
        if scale > self.cfg.drift_sigma_cap and self.pf_self_confidence_ema < self.cfg.drift_pf_confidence_threshold:
            scale = self.cfg.drift_sigma_cap
        sigma = base_sigma * _clamp(scale, self.cfg.sigma_min_scale, self.cfg.sigma_max_scale)
        # When static: widen sigma so fine matches inform but don't jerk the PF
        if is_static:
            sigma = max(sigma, base_sigma * 2.0)
        return sigma

    def get_effective_kappa(self, confidence: float, is_static: bool = False) -> float:
        kappa = 15.0 * confidence
        # When static: reduce heading pull (same image = noisy heading estimates)
        if is_static:
            kappa *= 0.3
        return kappa

    def evaluate_global_correction(
        self,
        recon_sim: float,
        coarse_fine_east: Optional[float],
        coarse_fine_north: Optional[float],
        coarse_fine_inliers: int,
        coarse_fine_heading: Optional[float],
        pf_east: float,
        pf_north: float,
        pf_spread: float,
    ) -> Optional[GlobalCorrectionResult]:
        """Check if a global correction should fire.

        Accumulates evidence across frames: when reconstruction consistently
        disagrees with the drone view but coarse+fine consistently finds a
        strong match elsewhere, triggers a hard particle teleport.
        """
        cfg = self.cfg
        if not cfg.global_corr_enabled:
            return None

        # Tick cooldown
        if self._global_correction_cooldown > 0:
            self._global_correction_cooldown -= 1

        # Accumulate or reset
        divergent = (
            recon_sim < cfg.global_corr_recon_sim_threshold
            and pf_spread < cfg.global_corr_max_spread_m
        )
        if divergent:
            self._recon_diverge_count += 1
            if (coarse_fine_east is not None
                    and coarse_fine_north is not None
                    and coarse_fine_inliers >= cfg.global_corr_min_inliers):
                self._coarse_fine_hits.append((
                    coarse_fine_east, coarse_fine_north,
                    coarse_fine_inliers, coarse_fine_heading,
                ))
        else:
            self._recon_diverge_count = 0
            self._coarse_fine_hits.clear()

        # Check trigger conditions
        if (self._recon_diverge_count < cfg.global_corr_min_frames
                or len(self._coarse_fine_hits) < cfg.global_corr_min_frames
                or self._global_correction_cooldown > 0):
            return None

        # Compute median position from accumulated hits
        hits = self._coarse_fine_hits
        easts = [h[0] for h in hits]
        norths = [h[1] for h in hits]
        inliers_list = [h[2] for h in hits]
        med_e = float(np.median(easts))
        med_n = float(np.median(norths))

        # Spatial consistency: hit positions must cluster
        hit_spread = math.sqrt(np.var(easts) + np.var(norths))
        if hit_spread > cfg.global_corr_position_consistency_m:
            return None

        # Distance from PF: must be significantly far (confirms PF is wrong)
        dist_from_pf = math.sqrt((med_e - pf_east) ** 2 + (med_n - pf_north) ** 2)
        if dist_from_pf < pf_spread * 2.0:
            return None

        # Heading: circular median, reject if too noisy
        headings = [h[3] for h in hits if h[3] is not None]
        result_heading: Optional[float] = None
        if headings:
            rads = [math.radians(h) for h in headings]
            sin_sum = sum(math.sin(r) for r in rads)
            cos_sum = sum(math.cos(r) for r in rads)
            mean_hdg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
            # Circular std: R = resultant length / n
            R = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(rads)
            circ_std_deg = math.degrees(math.sqrt(-2.0 * math.log(max(R, 1e-6))))
            if circ_std_deg < 30.0:
                result_heading = mean_hdg

        # Fire correction
        self._global_correction_cooldown = cfg.global_corr_cooldown_frames
        self._recon_diverge_count = 0
        self._coarse_fine_hits.clear()

        return GlobalCorrectionResult(
            east_m=med_e,
            north_m=med_n,
            heading_deg=result_heading,
            inliers_median=int(np.median(inliers_list)),
            n_consistent_frames=len(hits),
            distance_from_pf_m=dist_from_pf,
        )


def evaluate_coarse_trust(
    top1_sim: float,
    pf_spread: float,
    recon_sim_ema: float,
    cfg: TrustConfig,
) -> Optional[Tuple[float, float]]:
    """Decide fraction of particles to teleport and spread.

    Returns (teleport_fraction, teleport_sigma) or None if no teleport.
    """
    # Base trust from similarity
    denom = cfg.coarse_sim_ceiling - cfg.coarse_sim_floor
    if denom <= 0:
        sim_conf = 1.0 if top1_sim >= cfg.coarse_sim_floor else 0.0
    else:
        sim_conf = _clamp((top1_sim - cfg.coarse_sim_floor) / denom, 0.0, 1.0)

    # Boost when PF is uncertain (spread large = more willing to teleport)
    spread_boost = _clamp(pf_spread / 100.0, 0.0, 1.0)

    # Suppress when recon_sim is high (mosaic agrees with PF, no need to teleport)
    recon_suppress = 1.0 - _clamp(recon_sim_ema - 0.3, 0.0, 0.5) * 2.0

    coarse_confidence = sim_conf * (0.5 + 0.5 * spread_boost) * recon_suppress

    if coarse_confidence < 0.15:
        return None

    # Continuous fraction & spread
    teleport_fraction = _clamp(
        coarse_confidence * 0.8, 0.1, cfg.coarse_max_teleport_fraction
    )
    sigma_lo, sigma_hi = cfg.coarse_teleport_sigma_range
    teleport_sigma = sigma_hi / (1.0 + coarse_confidence)
    teleport_sigma = _clamp(teleport_sigma, sigma_lo, sigma_hi)

    return teleport_fraction, teleport_sigma
