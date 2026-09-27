from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Point


@dataclass(frozen=True)
class HazardGrid:
    lats: np.ndarray
    lons: np.ndarray
    hazard: np.ndarray
    elevation: np.ndarray
    terrain_source: str
    storm_center: Point
    storm_radius_km: float
    rainfall_mm: float
    sample_area_km2: float

    def sample(self, point: Point) -> float:
        yi = int(np.clip(np.searchsorted(self.lats, point.lat), 1, len(self.lats) - 1))
        xi = int(np.clip(np.searchsorted(self.lons, point.lon), 1, len(self.lons) - 1))
        candidates = []
        for y, x in ((yi - 1, xi - 1), (yi - 1, xi), (yi, xi - 1), (yi, xi)):
            candidates.append((abs(self.lats[y] - point.lat) + abs(self.lons[x] - point.lon), y, x))
        _, y, x = min(candidates)
        return float(self.hazard[y, x])


def _km_per_degree_lon(lat: float) -> float:
    return max(40.0, 111.32 * np.cos(np.radians(lat)))


def build_hazard(
    a: Point,
    b: Point,
    rainfall_mm: float,
    duration_min: int,
    elevation_grid: np.ndarray,
    n: int = 31,
    storm_center: Point | None = None,
    storm_radius_km: float = 3.0,
) -> HazardGrid:
    center = storm_center or Point((a.lat + b.lat) / 2, (a.lon + b.lon) / 2)
    radius = float(max(0.5, storm_radius_km))

    radius_lat = radius / 111.32
    radius_lon = radius / _km_per_degree_lon(center.lat)
    min_lat = min(a.lat, b.lat, center.lat) - max(0.01, 1.35 * radius_lat)
    max_lat = max(a.lat, b.lat, center.lat) + max(0.01, 1.35 * radius_lat)
    min_lon = min(a.lon, b.lon, center.lon) - max(0.01, 1.35 * radius_lon)
    max_lon = max(a.lon, b.lon, center.lon) + max(0.01, 1.35 * radius_lon)
    if elevation_grid.ndim != 2 or elevation_grid.shape[0] != elevation_grid.shape[1]:
        raise ValueError("Elevation grid must be square")
    # Preserve caller-provided test grids; production uses a 31×31 grid.
    n = int(elevation_grid.shape[0])
    lats = np.linspace(min_lat, max_lat, n)
    lons = np.linspace(min_lon, max_lon, n)

    lon2, lat2 = np.meshgrid(lons, lats)
    # Great-circle-ish distance approximation sufficient for this prototype grid.
    dx_km = (lon2 - center.lon) * _km_per_degree_lon(center.lat)
    dy_km = (lat2 - center.lat) * 111.32
    dist_km = np.hypot(dx_km, dy_km)

    # Rain is strongest near the selected storm area and decays smoothly outside it.
    sigma = max(radius * 0.58, 0.5)
    rain_field = rainfall_mm * np.exp(-0.5 * (dist_km / sigma) ** 2)

    z = elevation_grid.astype(float)
    gy, gx = np.gradient(z)
    slope = np.hypot(gx, gy)
    slope_norm = np.clip(slope / (np.percentile(slope, 90) + 1e-9), 0, 1)

    low_elev = 1.0 - (z - np.min(z)) / (np.ptp(z) + 1e-9)
    local_mean = (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) + z
    ) / 5.0
    depression = np.clip((local_mean - z) / (np.std(z) + 1e-9), -1, 1)
    depression = (depression + 1) / 2

    # A terrain susceptibility proxy: low ground + local depression + gentle slope.
    terrain_factor = np.clip(
        0.50 * low_elev + 0.30 * depression + 0.20 * (1 - slope_norm), 0, 1
    )

    duration_factor = np.sqrt(np.clip(duration_min, 10, 180) / 60.0)
    accumulation_proxy = rain_field * terrain_factor * duration_factor
    # Smooth saturating response. This is a prototype hazard index, not calibrated water depth.
    hazard = 1.0 - np.exp(-accumulation_proxy / 36.0)
    hazard = np.clip(hazard, 0.0, 1.0)

    area = max(0.1, (max_lat - min_lat) * 111.32 * (max_lon - min_lon) * _km_per_degree_lon(center.lat))
    return HazardGrid(
        lats=lats,
        lons=lons,
        hazard=hazard,
        elevation=z,
        terrain_source="provided grid",
        storm_center=center,
        storm_radius_km=radius,
        rainfall_mm=float(rainfall_mm),
        sample_area_km2=float(area),
    )
