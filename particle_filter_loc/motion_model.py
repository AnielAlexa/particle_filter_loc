"""Motion models for the particle filter prediction step."""

import math
from dataclasses import dataclass
from typing import Optional

METERS_PER_DEGREE_LAT = 111320.0


@dataclass
class MotionDelta:
    dx_m: float        # east delta
    dy_m: float        # north delta
    heading_deg: float  # absolute heading
    dt_s: float


def get_relative_movement(
    lat_prev: float, lon_prev: float,
    lat_curr: float, lon_curr: float,
) -> tuple:
    delta_lat = lat_curr - lat_prev
    delta_lon = lon_curr - lon_prev
    lat_rad = math.radians(lat_prev)
    delta_y = delta_lat * METERS_PER_DEGREE_LAT
    delta_x = delta_lon * (METERS_PER_DEGREE_LAT * math.cos(lat_rad))
    return delta_x, delta_y


class MotionModelBase:
    def update(self, timestamp_ns: int, **kwargs) -> Optional[MotionDelta]:
        raise NotImplementedError


class RTKMotionModel(MotionModelBase):
    """Computes position deltas from consecutive RTK NavSatFix messages.
    Heading from separate yaw topic (Float64)."""

    def __init__(self):
        self._prev_lat: Optional[float] = None
        self._prev_lon: Optional[float] = None
        self._prev_ts: Optional[int] = None
        self._yaw_deg: float = 0.0

    def set_yaw(self, yaw_deg: float):
        self._yaw_deg = yaw_deg

    def update(self, timestamp_ns: int, lat: float = 0.0, lon: float = 0.0, **kwargs) -> Optional[MotionDelta]:
        if self._prev_lat is None:
            self._prev_lat = lat
            self._prev_lon = lon
            self._prev_ts = timestamp_ns
            return None

        dx, dy = get_relative_movement(self._prev_lat, self._prev_lon, lat, lon)
        dt_s = (timestamp_ns - self._prev_ts) * 1e-9

        self._prev_lat = lat
        self._prev_lon = lon
        self._prev_ts = timestamp_ns

        if dt_s <= 0:
            return None

        return MotionDelta(dx_m=dx, dy_m=dy, heading_deg=self._yaw_deg, dt_s=dt_s)
