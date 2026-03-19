"""Core particle filter for UAV geo-localization."""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from .motion_model import MotionDelta


class Phase(Enum):
    UNINIT = auto()
    DISPERSED = auto()
    CONVERGING = auto()
    TRACKING = auto()


@dataclass
class PFConfig:
    n_dispersed: int = 200
    n_tracking: int = 100
    sigma_pos_dispersed: float = 3.0
    sigma_pos_tracking: float = 1.0
    sigma_hdg_dispersed: float = 10.0
    sigma_hdg_tracking: float = 3.0
    sigma_obs_coarse: float = 30.0      # was 50 — tighter at >50m altitude
    sigma_obs_fine: float = 8.0         # was 5 — accounts for GPS metadata resolution
    top_k_coarse: int = 5
    ess_threshold_fraction: float = 0.5
    init_altitude_m: float = 50.0       # was 15 — only init when high enough for VPR
    converge_spread_m: float = 80.0
    tracking_spread_m: float = 20.0
    lost_spread_m: float = 150.0
    fine_every_n_frames: int = 3
    fine_min_inliers_heading: int = 12
    min_search_radius_m: float = 40.0
    search_radius_multiplier: float = 2.5
    base_context_fraction: float = 0.25
    max_context_fraction: float = 1.0
    # Altitude-adaptive coarse sigma: scale sigma_obs_coarse with altitude
    altitude_sigma_enabled: bool = False
    altitude_sigma_ref_m: float = 60.0   # reference altitude for sigma_obs_coarse
    altitude_sigma_scale: float = 0.5    # sigma *= clamp(alt/ref * scale, 0.5, 2.0)
    # Fine match consistency gate: reject fine updates too far from current estimate
    # Prevents visually-ambiguous patches from jumping the PF across the map.
    # Set to 0.0 to disable (always accept fine).
    fine_consistency_max_m: float = 0.0
    # Strong coarse trust: when top-1 sim exceeds this threshold, teleport most
    # particles to the matched patch center.  Set to 0.0 to disable.
    coarse_trust_sim: float = 0.0
    coarse_trust_fraction: float = 0.7   # fraction of particles to inject
    coarse_trust_sigma: float = 20.0     # spatial spread of injected particles (m)


class ParticleFilter:

    def __init__(self, config: PFConfig, rng_seed: int = 42):
        self.cfg = config
        self.rng = np.random.default_rng(rng_seed)
        self.phase = Phase.UNINIT
        self.particles: Optional[np.ndarray] = None  # [N, 3] east, north, heading
        self.weights: Optional[np.ndarray] = None     # [N]
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def try_init(self, altitude_m: float) -> bool:
        if self.phase != Phase.UNINIT:
            return True
        return altitude_m > self.cfg.init_altitude_m

    def seed_from_coarse(self, patch_centers_enu: List[Tuple[float, float]], similarities: List[float],
                         sigma_override: Optional[float] = None):
        """Seed particles as Gaussian blobs around top-K coarse matches."""
        sigma = sigma_override if sigma_override is not None else self.cfg.sigma_obs_coarse
        n = self.cfg.n_dispersed
        sims = np.array(similarities, dtype=np.float64)
        sims = np.clip(sims, 0, None)
        total = sims.sum()
        if total <= 0:
            sims = np.ones_like(sims)
            total = sims.sum()
        probs = sims / total

        # Allocate particles per center proportional to similarity
        counts = np.round(probs * n).astype(int)
        # Fix rounding to sum exactly to n
        diff = n - counts.sum()
        counts[np.argmax(counts)] += diff

        particles = []
        for (east, north), count in zip(patch_centers_enu, counts):
            if count <= 0:
                continue
            e = self.rng.normal(east, sigma, size=count)
            nn = self.rng.normal(north, sigma, size=count)
            h = self.rng.uniform(0, 360, size=count)
            particles.append(np.column_stack([e, nn, h]))

        self.particles = np.vstack(particles)
        self.weights = np.full(len(self.particles), 1.0 / len(self.particles))
        self.phase = Phase.DISPERSED
        self._frame_count = 0

    def seed_from_position(self, east: float, north: float, heading_deg: float = 0.0,
                           sigma_pos: float = 5.0, sigma_hdg: float = 10.0):
        """Seed particles tightly around a known position (e.g. RTK fix)."""
        n = self.cfg.n_dispersed
        e = self.rng.normal(east, sigma_pos, size=n)
        nn = self.rng.normal(north, sigma_pos, size=n)
        h = self.rng.normal(heading_deg, sigma_hdg, size=n) % 360.0
        self.particles = np.column_stack([e, nn, h])
        self.weights = np.full(n, 1.0 / n)
        self.phase = Phase.CONVERGING
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, delta: MotionDelta):
        if self.particles is None:
            return

        n = len(self.particles)
        if self.phase == Phase.TRACKING:
            sigma_pos = self.cfg.sigma_pos_tracking
            sigma_hdg = self.cfg.sigma_hdg_tracking
        else:
            sigma_pos = self.cfg.sigma_pos_dispersed
            sigma_hdg = self.cfg.sigma_hdg_dispersed

        self.particles[:, 0] += delta.dx_m + self.rng.normal(0, sigma_pos, n)
        self.particles[:, 1] += delta.dy_m + self.rng.normal(0, sigma_pos, n)
        self.particles[:, 2] = delta.heading_deg + self.rng.normal(0, sigma_hdg, n)
        # Normalize heading to [0, 360)
        self.particles[:, 2] %= 360.0

    # ------------------------------------------------------------------
    # Observation updates
    # ------------------------------------------------------------------

    def get_obs_sigma_coarse(self, altitude_m: float = 0.0) -> float:
        """Return coarse observation sigma, optionally scaled by altitude."""
        if not self.cfg.altitude_sigma_enabled or altitude_m <= 0.0:
            return self.cfg.sigma_obs_coarse
        # At higher altitude the drone footprint is larger → less precise patch center
        scale = (altitude_m / self.cfg.altitude_sigma_ref_m) * self.cfg.altitude_sigma_scale
        scale = max(0.5, min(scale, 2.0))
        return self.cfg.sigma_obs_coarse * scale

    def update_coarse(self, top_k_patches: List[Tuple[float, float, float]],
                      altitude_m: float = 0.0,
                      temperature: float = 0.05):
        """Update weights from coarse matches using a Gaussian mixture model.

        Each tuple: (east, north, similarity).
        The K patches are treated as K components of a single mixture observation:
            p(z | x_i) = sum_k  w_k * N(x_i; mu_k, sigma²)
        where w_k = softmax(sim_k / T) — sim scores select which patch to believe,
        not how much total evidence there is.  This avoids double-counting when
        multiple patches cluster together.
        """
        if self.particles is None or len(top_k_patches) == 0:
            return

        sims = np.array([s for _, _, s in top_k_patches], dtype=np.float64)
        # Softmax mixture weights over the K candidates
        log_mix = sims / temperature
        log_mix -= log_mix.max()
        mix_w = np.exp(log_mix)
        mix_w /= mix_w.sum()  # [K]

        sigma = self.get_obs_sigma_coarse(altitude_m)
        sigma2 = 2.0 * sigma ** 2
        N = len(self.particles)

        # log p(z | x_i) = logsumexp_k [ log(mix_w[k]) - dist(x_i, mu_k)^2 / sigma2 ]
        # Shape: [N, K]
        log_components = np.empty((N, len(top_k_patches)), dtype=np.float64)
        for k, (east, north, _) in enumerate(top_k_patches):
            dx = self.particles[:, 0] - east
            dy = self.particles[:, 1] - north
            log_components[:, k] = np.log(mix_w[k] + 1e-300) - (dx**2 + dy**2) / sigma2

        # logsumexp over K for each particle
        lse_max = log_components.max(axis=1, keepdims=True)
        log_likelihood = lse_max.squeeze(1) + np.log(
            np.exp(log_components - lse_max).sum(axis=1) + 1e-300
        )

        log_weights = np.log(self.weights + 1e-300) + log_likelihood
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.full(N, 1.0 / N)

    def inject_coarse_trust(self, east: float, north: float, sim: float):
        """If sim exceeds threshold, teleport most particles to (east, north).

        Replaces `coarse_trust_fraction` of particles with fresh samples
        drawn from a tight Gaussian around the match, keeping the rest
        for diversity.  Weights are reset to uniform.
        """
        if self.particles is None:
            return
        if self.cfg.coarse_trust_sim <= 0.0 or sim < self.cfg.coarse_trust_sim:
            return

        n = len(self.particles)
        n_inject = int(self.cfg.coarse_trust_fraction * n)
        n_keep = n - n_inject

        # Keep the highest-weight existing particles
        keep_idx = np.argsort(self.weights)[-n_keep:]

        # Preserve current heading estimate for injected particles
        est_hdg = float(np.average(self.particles[:, 2], weights=self.weights))

        sigma = self.cfg.coarse_trust_sigma
        new_e = self.rng.normal(east, sigma, n_inject)
        new_n = self.rng.normal(north, sigma, n_inject)
        new_h = self.rng.normal(est_hdg, 10.0, n_inject) % 360.0

        self.particles = np.vstack([
            self.particles[keep_idx],
            np.column_stack([new_e, new_n, new_h]),
        ])
        self.weights = np.full(n, 1.0 / n)

    def update_fine(self, fine_east: float, fine_north: float, inliers: int,
                    heading_deg: Optional[float] = None) -> bool:
        """Update weights from fine match result.

        Returns True if update was applied, False if rejected by consistency gate.
        """
        if self.particles is None:
            return False

        # Consistency gate: reject if fine match is too far from current estimate
        if self.cfg.fine_consistency_max_m > 0.0:
            est_e, est_n, _ = self.estimate()
            dist = math.sqrt((fine_east - est_e) ** 2 + (fine_north - est_n) ** 2)
            if dist > self.cfg.fine_consistency_max_m:
                return False

        sigma2 = 2.0 * self.cfg.sigma_obs_fine ** 2
        dx = self.particles[:, 0] - fine_east
        dy = self.particles[:, 1] - fine_north
        dist2 = dx ** 2 + dy ** 2
        log_likelihood = -dist2 / sigma2

        # Von Mises heading update if enough inliers
        if heading_deg is not None and inliers >= self.cfg.fine_min_inliers_heading:
            kappa = 5.0  # ~26 deg std dev
            diff_rad = np.radians(self.particles[:, 2] - heading_deg)
            log_likelihood += kappa * np.cos(diff_rad)

        log_weights = np.log(self.weights + 1e-300) + log_likelihood
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.full(len(self.weights), 1.0 / len(self.weights))
        return True

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample_if_needed(self):
        if self.particles is None:
            return
        ess = self.effective_sample_size()
        threshold = self.cfg.ess_threshold_fraction * len(self.particles)
        if ess < threshold:
            self._systematic_resample()

    def _systematic_resample(self):
        n = len(self.particles)
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # ensure exact
        positions = (self.rng.random() + np.arange(n)) / n
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices].copy()
        self.weights = np.full(n, 1.0 / n)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def check_transitions(self):
        if self.particles is None:
            return

        spread = self.weighted_spread()

        if self.phase == Phase.DISPERSED:
            if spread < self.cfg.converge_spread_m:
                self.phase = Phase.CONVERGING
        elif self.phase == Phase.CONVERGING:
            if spread < self.cfg.tracking_spread_m:
                self._reduce_particles(self.cfg.n_tracking)
                self.phase = Phase.TRACKING
                self._frame_count = 0
            elif spread > self.cfg.lost_spread_m:
                self.phase = Phase.DISPERSED
        elif self.phase == Phase.TRACKING:
            if spread > self.cfg.lost_spread_m:
                self._expand_particles(self.cfg.n_dispersed)
                self.phase = Phase.DISPERSED
                self._frame_count = 0

    def _reduce_particles(self, target_n: int):
        """Resample down to target_n particles."""
        if len(self.particles) <= target_n:
            return
        self._systematic_resample()
        indices = self.rng.choice(len(self.particles), size=target_n, replace=False)
        self.particles = self.particles[indices].copy()
        self.weights = np.full(target_n, 1.0 / target_n)

    def _expand_particles(self, target_n: int):
        """Expand particle count by duplicating with noise."""
        if len(self.particles) >= target_n:
            return
        n_add = target_n - len(self.particles)
        indices = self.rng.choice(len(self.particles), size=n_add, replace=True, p=self.weights)
        new_particles = self.particles[indices].copy()
        new_particles[:, 0] += self.rng.normal(0, self.cfg.sigma_pos_dispersed, n_add)
        new_particles[:, 1] += self.rng.normal(0, self.cfg.sigma_pos_dispersed, n_add)
        new_particles[:, 2] += self.rng.normal(0, self.cfg.sigma_hdg_dispersed, n_add)
        new_particles[:, 2] %= 360.0
        self.particles = np.vstack([self.particles, new_particles])
        self.weights = np.full(len(self.particles), 1.0 / len(self.particles))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def estimate(self) -> Tuple[float, float, float]:
        """Weighted mean estimate: (east, north, heading_deg)."""
        if self.particles is None:
            return 0.0, 0.0, 0.0
        east = np.average(self.particles[:, 0], weights=self.weights)
        north = np.average(self.particles[:, 1], weights=self.weights)
        # Circular mean for heading
        rad = np.radians(self.particles[:, 2])
        sin_mean = np.average(np.sin(rad), weights=self.weights)
        cos_mean = np.average(np.cos(rad), weights=self.weights)
        heading = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0
        return float(east), float(north), float(heading)

    def should_run_fine(self) -> bool:
        self._frame_count += 1
        if self.phase == Phase.DISPERSED:
            return False
        if self.phase == Phase.CONVERGING:
            return True
        # TRACKING: every N frames
        return (self._frame_count % self.cfg.fine_every_n_frames) == 0

    def effective_sample_size(self) -> float:
        if self.weights is None:
            return 0.0
        return 1.0 / (np.sum(self.weights ** 2) + 1e-300)

    def weighted_spread(self) -> float:
        """Weighted standard deviation of particle positions (meters)."""
        if self.particles is None:
            return float('inf')
        mean_e = np.average(self.particles[:, 0], weights=self.weights)
        mean_n = np.average(self.particles[:, 1], weights=self.weights)
        var_e = np.average((self.particles[:, 0] - mean_e) ** 2, weights=self.weights)
        var_n = np.average((self.particles[:, 1] - mean_n) ** 2, weights=self.weights)
        return float(np.sqrt(var_e + var_n))

    # ------------------------------------------------------------------
    # Adaptive search scope
    # ------------------------------------------------------------------

    def get_search_radius(self) -> float:
        spread = self.weighted_spread()
        return max(spread * self.cfg.search_radius_multiplier, self.cfg.min_search_radius_m)

    def get_fine_top_k(self) -> int:
        spread = self.weighted_spread()
        if spread > 50.0:
            return 3
        elif spread > 20.0:
            return 2
        return 1

    def get_context_fraction(self, base: float = 0.25) -> float:
        spread = self.weighted_spread()
        frac = base + spread / 200.0
        return float(np.clip(frac, self.cfg.base_context_fraction, self.cfg.max_context_fraction))
