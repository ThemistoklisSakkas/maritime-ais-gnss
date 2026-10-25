from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from maritime.ais.encounters import (
    STSParams,
    build_episodes,
    classify_pairs,
    close_pairs,
    drop_routine,
    flag_dark_context,
    pairs_within,
    slow_positions,
)
from maritime.zones import Zones

T0 = datetime(2025, 6, 3, tzinfo=UTC)
P = STSParams(bin_min=5, max_sog_kn=3.0, max_dist_m=500.0, min_duration_h=1.0,
              max_missing_bins=2, zone_cell_deg=0.005,
              exclude_ship_types=("Sailing", "Pleasure"),
              service_ship_types=("Tug", "Pilot", "SAR"), dark_window_h=12.0)
CRS = "EPSG:25832"
M_PER_DEG_LAT = 111_195.0


def stationary(mmsi, lat, lon, hours, sog=0.5, every_s=180, start_h=0.0):
    n = int(hours * 3600 / every_s) + 1
    return pl.DataFrame({
        "mmsi": [mmsi] * n,
        "ts": [T0 + timedelta(hours=start_h, seconds=i * every_s) for i in range(n)],
        "lat": [lat] * n, "lon": [lon] * n, "sog": [sog] * n,
    })


def episodes(points):
    cand = slow_positions(points, P)
    return build_episodes(close_pairs(cand, P, CRS), P)


# ---------------------------------------------------------------- KD-tree correctness

def test_kdtree_pairs_match_brute_force():
    rng = np.random.default_rng(42)
    x, y = rng.uniform(0, 20_000, (2, 800))
    got = {tuple(r) for r in pairs_within(x, y, 500.0)}
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    i, j = np.nonzero(np.triu(d <= 500.0, k=1))
    assert got == set(zip(i.tolist(), j.tolist(), strict=True))
    assert len(got) > 0


def test_pairs_within_tiny_inputs():
    assert pairs_within(np.array([1.0]), np.array([1.0]), 500).shape == (0, 2)


# ---------------------------------------------------------------- episodes

LAT, LON = 55.3, 14.6
DLAT_300M = 300 / M_PER_DEG_LAT


def test_two_ships_alongside_for_two_hours():
    pts = pl.concat([stationary(219_000_001, LAT, LON, 2.0),
                     stationary(219_000_002, LAT + DLAT_300M, LON, 2.0)])
    ep = episodes(pts)
    assert ep.height == 1
    r = ep.row(0, named=True)
    assert (r["mmsi_a"], r["mmsi_b"]) == (219_000_001, 219_000_002)
    assert r["duration_h"] == pytest.approx(2.0, abs=0.1)
    assert r["mean_dist_m"] == pytest.approx(300.0, rel=0.01)


def test_too_far_apart():
    pts = pl.concat([stationary(219_000_001, LAT, LON, 2.0),
                     stationary(219_000_002, LAT + 700 / M_PER_DEG_LAT, LON, 2.0)])
    assert episodes(pts).is_empty()


def test_too_short():
    pts = pl.concat([stationary(219_000_001, LAT, LON, 0.6),
                     stationary(219_000_002, LAT + DLAT_300M, LON, 0.6)])
    assert episodes(pts).is_empty()


def test_one_vessel_moving_is_not_an_encounter():
    pts = pl.concat([stationary(219_000_001, LAT, LON, 2.0),
                     stationary(219_000_002, LAT + DLAT_300M, LON, 2.0, sog=6.0)])
    assert episodes(pts).is_empty()


def test_short_reception_dropout_is_bridged_long_one_splits():
    a = stationary(219_000_001, LAT, LON, 3.0)
    b = stationary(219_000_002, LAT + DLAT_300M, LON, 3.0)
    t = pl.col("ts")
    short = b.filter(~t.is_between(T0 + timedelta(minutes=60), T0 + timedelta(minutes=70)))
    assert episodes(pl.concat([a, short])).height == 1          # 10 min hole: bridged
    long = b.filter(~t.is_between(T0 + timedelta(minutes=60), T0 + timedelta(minutes=100)))
    eps = episodes(pl.concat([a, long]))
    assert eps.height == 2                                        # 40 min hole: two episodes
    assert (eps["duration_h"] >= 1.0).all()


def test_pair_order_is_canonical():
    pts = pl.concat([stationary(219_000_009, LAT, LON, 2.0),
                     stationary(219_000_001, LAT + DLAT_300M, LON, 2.0)])
    r = episodes(pts).row(0, named=True)
    assert r["mmsi_a"] < r["mmsi_b"]


# ---------------------------------------------------------------- filtering & context

def registry(types):
    return pl.DataFrame({"mmsi": list(types), "name": [f"V{m}" for m in types],
                         "ship_type": list(types.values()), "length": [100.0] * len(types)})


def test_drop_routine_zone_and_type():
    cand = pl.DataFrame({
        "mmsi": [1, 2, 3], "bin": [T0] * 3,
        "lat": [55.5, 56.5, 57.5], "lon": [11.5, 11.5, 11.5], "sog": [0.0] * 3,
    })
    ports = pl.DataFrame({"lat": [55.5], "lon": [11.5], "radius_km": [3.0]})
    zones = Zones(ports, pl.DataFrame(schema={"cx": pl.Int32, "cy": pl.Int32}), 0.02)
    reg = registry({1: "Cargo", 2: "Sailing", 3: "Tanker"})
    out, stats = drop_routine(cand, zones, reg, P)
    assert out["mmsi"].to_list() == [3]
    assert stats == {"slow_vessel_bins": 3, "removed_in_zone": 1, "removed_by_type": 1,
                     "candidate_bins": 1}


@pytest.mark.parametrize("ta, tb, expected", [
    ("Cargo", "Tanker", "cargo_tanker_pair"),
    ("Tanker", None, "cargo_tanker_with_unknown"),
    ("Tanker", "Tug", "service_involved"),
    ("Fishing", "Fishing", "fishing_pair"),
    ("Passenger", "Other", "other"),
])
def test_classify_pairs(ta, tb, expected):
    ep = pl.DataFrame({"mmsi_a": [1], "mmsi_b": [2]})
    reg = registry({1: ta, 2: tb}) if tb else registry({1: ta})
    out = classify_pairs(ep, reg, ["Cargo", "Tanker"], P)
    assert out["pair_class"].to_list() == [expected]


def test_dark_context_window():
    ep = pl.DataFrame({"mmsi_a": [1, 3], "mmsi_b": [2, 4],
                       "start": [T0, T0], "end": [T0 + timedelta(hours=2)] * 2})
    gaps = pl.DataFrame({
        "mmsi": [2, 4, 4], "category": ["dark", "dark", "low_coverage"],
        "gap_start": [T0 - timedelta(hours=8), T0 + timedelta(hours=20), T0],
        "gap_end": [T0 - timedelta(hours=4), T0 + timedelta(hours=24), T0 + timedelta(hours=1)],
    })
    out = flag_dark_context(ep, gaps, window_h=12)
    # vessel 2 went dark 4 h before: flagged. Vessel 4: dark gap 18 h after (outside window),
    # and the overlapping gap is low_coverage, not dark: not flagged.
    assert out["dark_nearby"].to_list() == [True, False]
