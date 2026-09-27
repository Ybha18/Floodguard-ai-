from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import requests

from .config import SETTINGS
from .models import Point


def fetch_elevation(points: Iterable[Point], timeout: float = 12.0) -> tuple[np.ndarray, str]:
    pts = list(points)
    if not pts:
        return np.array([], dtype=float), "none"
    vals: list[float] = []
    try:
        # Batch requests to avoid oversized URLs for a grid.
        for i in range(0, len(pts), 100):
            batch = pts[i:i + 100]
            params = {
                "latitude": ",".join(f"{p.lat:.6f}" for p in batch),
                "longitude": ",".join(f"{p.lon:.6f}" for p in batch),
            }
            r = requests.get(SETTINGS.open_meteo_elevation_url, params=params, timeout=timeout)
            r.raise_for_status()
            part = r.json().get("elevation", [])
            if len(part) != len(batch):
                raise ValueError("Elevation response length mismatch")
            vals.extend(float(x) for x in part)
        return np.asarray(vals, dtype=float), "Open-Meteo elevation (Copernicus GLO-90)"
    except Exception:
        # Smooth deterministic fallback, explicitly labelled as a proxy rather than observed terrain.
        out = []
        for p in pts:
            x = math.radians(p.lon)
            y = math.radians(p.lat)
            base = 55 * math.sin(6 * x) * math.cos(5 * y)
            secondary = 18 * math.sin(15 * x + 0.4) * math.cos(11 * y)
            out.append(120 + base + secondary)
        return np.asarray(out, dtype=float), "deterministic terrain proxy (offline fallback)"
