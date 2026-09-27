from __future__ import annotations

import math
from typing import Any

import requests

from .config import SETTINGS
from .hazard import HazardGrid
from .models import Point, RouteCandidate


def _route_query(url: str, points: list[Point], timeout: float) -> list[dict[str, Any]]:
    coordinates = ";".join(f"{p.lon},{p.lat}" for p in points)
    endpoint = f"{url}/route/v1/driving/{coordinates}"
    params = {
        "alternatives": "true",
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
        "annotations": "true",
    }
    r = requests.get(endpoint, params=params, timeout=timeout)
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    if data.get("code") != "Ok":
        return []
    return list(data.get("routes", []))


def _candidate_waypoints(a: Point, b: Point, center: Point, radius_km: float) -> list[tuple[str, list[Point]]]:
    # Real-road detours are produced by OSRM through explicit via points around the selected flood area.
    dlat = (radius_km * 1.55) / 111.32
    dlon = (radius_km * 1.55) / max(40.0, 111.32 * math.cos(math.radians(center.lat)))
    return [
        ("Direct", [a, b]),
        ("North detour", [a, Point(center.lat + dlat, center.lon), b]),
        ("South detour", [a, Point(center.lat - dlat, center.lon), b]),
        ("East detour", [a, Point(center.lat, center.lon + dlon), b]),
        ("West detour", [a, Point(center.lat, center.lon - dlon), b]),
    ]


def route_candidates(a: Point, b: Point, hazard: HazardGrid, objective: str = "Balanced", timeout: float = 12.0) -> list[RouteCandidate]:
    seen: set[tuple[int, int]] = set()
    out: list[RouteCandidate] = []
    requests_plan = _candidate_waypoints(a, b, hazard.storm_center, hazard.storm_radius_km)

    last_error = None
    for label, points in requests_plan:
        for base_url in SETTINGS.osrm_urls:
            try:
                routes = _route_query(base_url, points, timeout)
                provider_label = f"OSRM / OpenStreetMap ({base_url.split('//')[-1]})"
                for route in routes:
                    coords = route.get("geometry", {}).get("coordinates", [])
                    geom = [Point(float(c[1]), float(c[0])) for c in coords if len(c) >= 2]
                    if len(geom) < 2:
                        continue
                    key = (round(float(route.get("distance", 0))), round(float(route.get("duration", 0))))
                    if key in seen:
                        continue
                    seen.add(key)
                    exposure, blocked = route_hazard_stats(geom, hazard)
                    base_minutes = float(route.get("duration", 0)) / 60.0
                    if objective == "Safety":
                        score = 0.7 * base_minutes + 260.0 * exposure + 900.0 * blocked
                    elif objective == "Speed":
                        score = 1.3 * base_minutes + 70.0 * exposure + 180.0 * blocked
                    else:
                        score = base_minutes + 150.0 * exposure + 360.0 * blocked
                    out.append(RouteCandidate(
                        provider=provider_label,
                        distance_m=float(route.get("distance", 0)),
                        duration_s=float(route.get("duration", 0)),
                        geometry=geom,
                        flood_exposure=exposure,
                        blocked_fraction=blocked,
                        score=score,
                        summary=f"{label}; exposure {exposure:.0%}; blocked-share {blocked:.0%}",
                        raw=route,
                    ))
                if out:
                    break
            except Exception as exc:
                last_error = str(exc)
        # Continue to other detour plans even after one succeeds so we can compare options.

    # Keep the most distinct routes, prioritizing lower score while preserving geometry differences.
    out.sort(key=lambda r: r.score)
    return out[:6]


def route_hazard_stats(geometry: list[Point], hazard: HazardGrid, step: int = 4) -> tuple[float, float]:
    if not geometry:
        return 0.0, 1.0
    sampled = geometry[::max(1, step)]
    vals = [hazard.sample(p) for p in sampled]
    exposure = sum(vals) / max(1, len(vals))
    blocked = sum(v >= 0.70 for v in vals) / max(1, len(vals))
    return float(exposure), float(blocked)


def haversine_m(a: Point, b: Point) -> float:
    r = 6371000.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = math.radians(b.lat - a.lat)
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))
