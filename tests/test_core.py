from floodguard.hazard import build_hazard
from floodguard.models import Point


def test_hazard_is_deterministic():
    a, b = Point(36.80, 10.18), Point(36.82, 10.21)
    import numpy as np

    z = np.ones((21, 21)) * 30
    h1 = build_hazard(a, b, 70, 60, z)
    h2 = build_hazard(a, b, 70, 60, z)
    assert np.allclose(h1.hazard, h2.hazard)


def test_more_rain_increases_or_equal_hazard():
    a, b = Point(36.80, 10.18), Point(36.82, 10.21)
    import numpy as np

    z = np.ones((21, 21)) * 30
    low = build_hazard(a, b, 30, 60, z).hazard
    high = build_hazard(a, b, 100, 60, z).hazard
    assert high.mean() >= low.mean()


def test_points_are_distinct():
    a, b = Point(36.80, 10.18), Point(36.82, 10.21)
    assert (a.lat, a.lon) != (b.lat, b.lon)
