from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np

from .elevation import fetch_elevation
from .hazard import build_hazard
from .models import Point, SimulationResult, WeatherObservation
from .routing import route_candidates
from .weather import get_live_weather


def simulate(
    a: Point,
    b: Point,
    rainfall_mm: float,
    duration_min: int = 60,
    seed: int = 42,
    use_live_weather: bool = False,
    storm_center: Point | None = None,
    storm_radius_km: float = 3.0,
    objective: str = "Balanced",
) -> tuple[SimulationResult, object]:
    center = storm_center or Point((a.lat + b.lat) / 2, (a.lon + b.lon) / 2)
    run_key = (
        f"{a.lat:.6f}|{a.lon:.6f}|{b.lat:.6f}|{b.lon:.6f}|{center.lat:.6f}|{center.lon:.6f}|"
        f"{rainfall_mm:.2f}|{duration_min}|{storm_radius_km:.2f}|{seed}"
    )
    run_id = hashlib.sha1(run_key.encode()).hexdigest()[:12]

    # Hazard extent is driven by A/B plus the operator-selected storm area.
    radius_lat = storm_radius_km / 111.32
    radius_lon = storm_radius_km / max(40.0, 111.32 * np.cos(np.radians(center.lat)))
    min_lat = min(a.lat, b.lat, center.lat) - max(0.01, 1.35 * radius_lat)
    max_lat = max(a.lat, b.lat, center.lat) + max(0.01, 1.35 * radius_lat)
    min_lon = min(a.lon, b.lon, center.lon) - max(0.01, 1.35 * radius_lon)
    max_lon = max(a.lon, b.lon, center.lon) + max(0.01, 1.35 * radius_lon)
    n = 31
    lats = np.linspace(min_lat, max_lat, n)
    lons = np.linspace(min_lon, max_lon, n)
    pts = [Point(float(la), float(lo)) for la in lats for lo in lons]
    elev, terrain_source = fetch_elevation(pts)
    elev_grid = elev.reshape(n, n)

    live_weather: WeatherObservation | None = None
    effective_rainfall = float(rainfall_mm)
    if use_live_weather:
        try:
            live_weather = get_live_weather(center)
            observed = live_weather.rain_mm if live_weather.rain_mm is not None else live_weather.precipitation_mm
            if observed is not None and observed > 0:
                # Explicit prototype rule: observed current-hour rain is scaled to the selected duration
                # and used as a floor for the scenario forcing. No hidden substitution.
                effective_rainfall = max(effective_rainfall, float(observed) * duration_min / 60.0)
        except Exception:
            live_weather = None

    hazard = build_hazard(
        a,
        b,
        effective_rainfall,
        duration_min,
        elev_grid,
        n=n,
        storm_center=center,
        storm_radius_km=storm_radius_km,
    )
    routes = route_candidates(a, b, hazard, objective=objective)

    viable = [i for i, r in enumerate(routes) if r.blocked_fraction < 0.35]
    rec = min(viable, key=lambda i: routes[i].score) if viable else None

    affected = float(np.mean(hazard.hazard >= 0.35))
    high = float(np.mean(hazard.hazard >= 0.55))
    blocked = float(np.mean(hazard.hazard >= 0.70))
    exposure_index = float(np.clip(affected * (0.4 + effective_rainfall / 180.0), 0, 1))

    evidence = [
        f"Rainfall scenario: {rainfall_mm:.0f} mm over {duration_min} min; effective forcing {effective_rainfall:.1f} mm.",
        f"Flood area: center {center.lat:.5f}, {center.lon:.5f}; radius {storm_radius_km:.1f} km.",
        f"Hazard engine: localized rainfall × terrain susceptibility proxy; deterministic seed {seed}.",
        f"Terrain source: {terrain_source}.",
        f"Affected hazard cells: {affected:.0%}; high-risk cells: {high:.0%}; severe proxy: {blocked:.0%}.",
        "Routing: real road geometry from OSRM/OpenStreetMap; multiple detour candidates are evaluated against the simulated hazard field.",
        "Limit: route ranking is a prototype heuristic; it is not a certified emergency routing service.",
    ]
    if use_live_weather:
        if live_weather:
            evidence.append(
                f"Open-Meteo live observation at {live_weather.observed_at}: rain={live_weather.rain_mm} mm, precipitation={live_weather.precipitation_mm} mm."
            )
        else:
            evidence.append("Open-Meteo was requested but unavailable; the simulation remained scenario-driven.")

    result = SimulationResult(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        scenario_mm=float(rainfall_mm),
        effective_rainfall_mm=float(effective_rainfall),
        duration_min=int(duration_min),
        seed=int(seed),
        a=a,
        b=b,
        storm_center=center,
        storm_radius_km=float(storm_radius_km),
        max_hazard=float(hazard.hazard.max()),
        affected_fraction=affected,
        road_high_fraction=high,
        road_blocked_fraction=blocked,
        exposure_index=exposure_index,
        terrain_source=terrain_source,
        route_candidates=routes,
        recommended_route_index=rec,
        live_weather=live_weather,
        evidence=evidence,
    )
    return result, hazard
