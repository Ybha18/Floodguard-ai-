from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str | None = os.getenv("NVIDIA_API_KEY") or None
    # Fast workhorse for the first-pass reasoning/sub-agent role.
    nvidia_primary_model: str = os.getenv("NVIDIA_PRIMARY_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    # Final cross-check/synthesis model. Backward-compatible fallback keeps the
    # previous NVIDIA_REASONER_MODEL variable working.
    nvidia_synthesis_model: str = os.getenv(
        "NVIDIA_SYNTHESIS_MODEL",
        os.getenv("NVIDIA_REASONER_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
    )
    nvidia_vision_model: str = os.getenv("NVIDIA_VISION_MODEL", "meta/muse-glimmer-30b")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    enable_gemini: bool = os.getenv("ENABLE_GEMINI", "true").lower() == "true"
    enable_muse: bool = os.getenv("ENABLE_MUSE", "true").lower() == "true"
    osrm_urls: tuple[str, ...] = tuple(
        x.strip().rstrip("/")
        for x in os.getenv(
            "OSRM_URLS",
            "https://router.project-osrm.org;https://routing.openstreetmap.de/routed-car",
        ).split(";")
        if x.strip()
    )
    open_meteo_url: str = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1")
    open_meteo_elevation_url: str = os.getenv(
        "OPEN_METEO_ELEVATION_URL", "https://api.open-meteo.com/v1/elevation"
    )


SETTINGS = Settings()
