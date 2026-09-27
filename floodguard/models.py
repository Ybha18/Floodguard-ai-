from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Point:
    lat: float
    lon: float


@dataclass(frozen=True)
class RouteCandidate:
    provider: str
    distance_m: float
    duration_s: float
    geometry: list[Point]
    flood_exposure: float
    blocked_fraction: float
    score: float
    summary: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeatherObservation:
    provider: str
    latitude: float
    longitude: float
    observed_at: str
    precipitation_mm: float | None
    rain_mm: float | None
    temperature_c: float | None
    wind_kmh: float | None
    weather_code: int | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    run_id: str
    created_at: str
    scenario_mm: float
    effective_rainfall_mm: float
    duration_min: int
    seed: int
    a: Point
    b: Point
    storm_center: Point
    storm_radius_km: float
    max_hazard: float
    affected_fraction: float
    road_high_fraction: float
    road_blocked_fraction: float
    exposure_index: float
    terrain_source: str
    route_candidates: list[RouteCandidate]
    recommended_route_index: int | None
    live_weather: WeatherObservation | None
    evidence: list[str]
