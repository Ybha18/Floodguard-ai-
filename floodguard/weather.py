from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from .config import SETTINGS
from .models import Point, WeatherObservation


def get_live_weather(point: Point, timeout: float = 8.0) -> WeatherObservation:
    params = {
        "latitude": point.lat,
        "longitude": point.lon,
        "current": "temperature_2m,precipitation,rain,wind_speed_10m,weather_code",
        "timezone": "auto",
    }
    r = requests.get(f"{SETTINGS.open_meteo_url}/forecast", params=params, timeout=timeout)
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    cur = data.get("current", {})
    return WeatherObservation(
        provider="Open-Meteo",
        latitude=point.lat,
        longitude=point.lon,
        observed_at=cur.get("time", datetime.now(timezone.utc).isoformat()),
        precipitation_mm=_num(cur.get("precipitation")),
        rain_mm=_num(cur.get("rain")),
        temperature_c=_num(cur.get("temperature_2m")),
        wind_kmh=_num(cur.get("wind_speed_10m")),
        weather_code=_int(cur.get("weather_code")),
        raw=data,
    )


def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
