"""Geodetic helpers shared by Part A and Part B.

All functions are vectorised over numpy arrays. Angles in degrees at the API,
distances in metres, speeds in m/s unless the name says otherwise (e.g. _kn).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # IUGG mean Earth radius
KNOT_MS = 1852.0 / 3600.0     # 1 knot in m/s (1 NM = 1852 m exactly)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance on a spherical Earth.

    Spherical error vs. the WGS84 ellipsoid is < 0.5 %, negligible for
    gap/speed screening at AIS accuracy (~10 m).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_expr(lat1, lon1, lat2, lon2):
    """Polars-expression version of haversine_m (same formula, runs inside the query engine)."""
    import polars as pl

    lat1, lon1, lat2, lon2 = (e.radians() for e in (lat1, lon1, lat2, lon2))
    a = ((lat2 - lat1) / 2).sin() ** 2 + lat1.cos() * lat2.cos() * ((lon2 - lon1) / 2).sin() ** 2
    return 2 * EARTH_RADIUS_M * a.clip(0.0, 1.0).sqrt().arcsin().cast(pl.Float64)


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial true bearing from point 1 to point 2, in [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360.0


def angle_diff_deg(a, b):
    """Signed smallest difference a - b, wrapped to [-180, 180).

    Needed for course changes: 359 deg -> 1 deg is a 2 deg turn, not 358.
    """
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def angle_diff_expr(a, b):
    """Polars-expression version of angle_diff_deg.

    The double modulo makes the result independent of how the engine signs a
    negative remainder.
    """
    return ((a - b + 180.0) % 360.0 + 360.0) % 360.0 - 180.0


@lru_cache(maxsize=8)
def _transformer(crs_from: str, crs_to: str):
    from pyproj import Transformer

    return Transformer.from_crs(crs_from, crs_to, always_xy=True)


def to_metric(lon, lat, crs: str = "EPSG:25832"):
    """Project WGS84 lon/lat to a metric CRS (default UTM 32N). Returns (x, y) in metres."""
    return _transformer("EPSG:4326", crs).transform(lon, lat)


def from_metric(x, y, crs: str = "EPSG:25832"):
    """Inverse of to_metric. Returns (lon, lat)."""
    return _transformer(crs, "EPSG:4326").transform(x, y)
