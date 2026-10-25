import numpy as np
import pytest

from maritime.geo import (
    EARTH_RADIUS_M,
    KNOT_MS,
    angle_diff_deg,
    from_metric,
    haversine_m,
    initial_bearing_deg,
    to_metric,
)


def test_haversine_one_degree_latitude():
    expected = EARTH_RADIUS_M * np.pi / 180  # ~111.195 km
    assert haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(expected, rel=1e-9)


def test_haversine_one_arcminute_is_about_one_nautical_mile():
    # The nautical mile was defined from 1' of latitude; spherical model gives ~1853 m
    assert haversine_m(55.0, 12.0, 55.0 + 1 / 60, 12.0) == pytest.approx(1852.0, rel=2e-3)


def test_haversine_zero_and_vectorised():
    lat = np.array([55.0, 56.0])
    lon = np.array([12.0, 13.0])
    d = haversine_m(lat, lon, lat, lon)
    assert d.shape == (2,)
    assert np.allclose(d, 0.0)


def test_polars_haversine_matches_numpy():
    import polars as pl

    from maritime.geo import haversine_expr

    rng = np.random.default_rng(0)
    lat1, lat2 = rng.uniform(53, 59, (2, 100))
    lon1, lon2 = rng.uniform(3, 17, (2, 100))
    df = pl.DataFrame({"a": lat1, "b": lon1, "c": lat2, "d": lon2})
    d_pl = df.select(haversine_expr(pl.col("a"), pl.col("b"), pl.col("c"), pl.col("d")))
    assert np.allclose(d_pl.to_series().to_numpy(), haversine_m(lat1, lon1, lat2, lon2))


@pytest.mark.parametrize(
    "lat2, lon2, expected",
    [(1.0, 0.0, 0.0), (0.0, 1.0, 90.0), (-1.0, 0.0, 180.0), (0.0, -1.0, 270.0)],
)
def test_initial_bearing_cardinal(lat2, lon2, expected):
    assert initial_bearing_deg(0.0, 0.0, lat2, lon2) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "a, b, expected",
    [(1.0, 359.0, 2.0), (359.0, 1.0, -2.0), (90.0, 0.0, 90.0), (180.0, 0.0, -180.0)],
)
def test_angle_diff_wraps(a, b, expected):
    assert angle_diff_deg(a, b) == pytest.approx(expected)


def test_knot_conversion():
    assert 10 * KNOT_MS == pytest.approx(5.1444, abs=1e-4)


def test_utm32_central_meridian_false_easting():
    x, y = to_metric(9.0, 0.0)  # zone 32 central meridian, equator
    assert x == pytest.approx(500_000.0, abs=1e-3)
    assert y == pytest.approx(0.0, abs=1e-3)


def test_utm_round_trip_and_metric_distance():
    lon = np.array([12.0, 12.0])
    lat = np.array([55.0, 55.0 + 1 / 60])
    x, y = to_metric(lon, lat)
    lon_back, lat_back = from_metric(x, y)
    assert np.allclose(lon_back, lon, atol=1e-9)
    assert np.allclose(lat_back, lat, atol=1e-9)
    # UTM is ellipsoidal, haversine is spherical. At 55N the WGS84 meridional radius
    # (~6378 km) exceeds the mean radius (6371 km) by ~0.12 %, so allow 0.5 %.
    d_proj = np.hypot(x[1] - x[0], y[1] - y[0])
    assert d_proj == pytest.approx(haversine_m(lat[0], lon[0], lat[1], lon[1]), rel=5e-3)
