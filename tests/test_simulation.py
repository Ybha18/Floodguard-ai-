import numpy as np

from floodguard.models import Point, RouteCandidate, WeatherObservation
from floodguard.simulation import simulate


def test_selected_flood_area_changes_hazard(monkeypatch):
    def fake_elevation(points):
        return np.linspace(30, 90, len(points)), "test terrain"

    def fake_routes(a, b, hazard, objective="Balanced"):
        return []

    monkeypatch.setattr("floodguard.simulation.fetch_elevation", fake_elevation)
    monkeypatch.setattr("floodguard.simulation.route_candidates", fake_routes)

    a = Point(36.80, 10.18)
    b = Point(36.82, 10.21)
    near = Point(36.81, 10.195)
    far = Point(36.95, 10.40)
    _, h_near = simulate(a, b, 100, storm_center=near, storm_radius_km=3)
    _, h_far = simulate(a, b, 100, storm_center=far, storm_radius_km=3)
    assert not np.allclose(h_near.hazard, h_far.hazard)


def test_live_weather_can_raise_effective_forcing(monkeypatch):
    def fake_elevation(points):
        return np.ones(len(points)) * 40, "test terrain"

    def fake_routes(a, b, hazard, objective="Balanced"):
        return []

    def fake_weather(point):
        return WeatherObservation(
            provider="Open-Meteo",
            latitude=point.lat,
            longitude=point.lon,
            observed_at="test",
            precipitation_mm=80.0,
            rain_mm=80.0,
            temperature_c=25.0,
            wind_kmh=10.0,
            weather_code=61,
            raw={},
        )

    monkeypatch.setattr("floodguard.simulation.fetch_elevation", fake_elevation)
    monkeypatch.setattr("floodguard.simulation.route_candidates", fake_routes)
    monkeypatch.setattr("floodguard.simulation.get_live_weather", fake_weather)

    a = Point(36.80, 10.18)
    b = Point(36.82, 10.21)
    result, _ = simulate(a, b, 30, duration_min=60, use_live_weather=True)
    assert result.effective_rainfall_mm == 80.0
