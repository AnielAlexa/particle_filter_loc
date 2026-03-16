"""WGS84 <-> ENU coordinate conversions."""

import math
from typing import Tuple

METERS_PER_DEG = 111319.5


class ENUFrame:
    """Local ENU frame anchored at a WGS84 origin."""

    def __init__(self, origin_lat: float, origin_lon: float):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self._cos_lat = math.cos(math.radians(origin_lat))

    def wgs84_to_enu(self, lat: float, lon: float) -> Tuple[float, float]:
        east = (lon - self.origin_lon) * METERS_PER_DEG * self._cos_lat
        north = (lat - self.origin_lat) * METERS_PER_DEG
        return east, north

    def enu_to_wgs84(self, east: float, north: float) -> Tuple[float, float]:
        lat = self.origin_lat + north / METERS_PER_DEG
        lon = self.origin_lon + east / (METERS_PER_DEG * self._cos_lat)
        return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def normalize_angle(deg: float) -> float:
    """Normalize angle to [-180, 180)."""
    deg = deg % 360.0
    if deg >= 180.0:
        deg -= 360.0
    return deg
