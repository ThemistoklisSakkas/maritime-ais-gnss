from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from maritime.ais.dark import (
    DarkParams,
    ManeuverParams,
    add_coverage,
    classify_gaps,
    detect_maneuvers,
    find_gaps,
)
from maritime.geo import KNOT_MS
from maritime.zones import Zones

T0 = datetime(2025, 6, 2, tzinfo=UTC)
DP = DarkParams(min_gap_h=2.0, max_gap_speed_kn=30.0, coverage_cell_deg=0.1,
                coverage_min_vessels=3, coverage_min_fraction=0.8, min_move_nm=1.0)
MP = ManeuverParams(window_min=10, min_turn_deg=150, turn_min_sog_kn=5,
                    stop_from_kn=8, stop_to_kn=2, merge_min=30)
M_PER_DEG_LAT = 111_195.0


def h(x):
    return T0 + timedelta(hours=x)


def segments(rows):
    """rows: (seg, start_h, end_h, start_lat, end_lat) for one vessel at lon 14.05."""
    return pl.DataFrame({
        "mmsi": [219_000_001] * len(rows),
        "seg": [r[0] for r in rows],
        "start": [h(r[1]) for r in rows], "end": [h(r[2]) for r in rows],
        "start_lat": [r[3] for r in rows], "end_lat": [r[4] for r in rows],
        "start_lon": [14.05] * len(rows), "end_lon": [14.05] * len(rows),
        "start_sog_kn": [10.0] * len(rows), "end_sog_kn": [10.0] * len(rows),
    })


def no_zones():
    return Zones(pl.DataFrame({"lat": [], "lon": [], "radius_km": []}),
                 pl.DataFrame(schema={"cx": pl.Int32, "cy": pl.Int32}), 0.02)


# ---------------------------------------------------------------- gaps

def test_find_gaps_threshold_and_speed():
    seg = segments([(0, 0, 1, 55.0, 55.1), (1, 4, 5, 55.6, 55.7), (2, 6, 7, 55.7, 55.8)])
    g = find_gaps(seg, min_gap_h=2.0)
    assert g.height == 1                              # the 1 h silence is ignored
    r = g.row(0, named=True)
    assert r["gap_h"] == pytest.approx(3.0)
    assert r["gap_dist_nm"] == pytest.approx(30.0, rel=1e-2)   # 0.5 deg lat = 30 NM
    assert r["implied_speed_kn"] == pytest.approx(10.0, rel=1e-2)


def test_no_gap_between_different_vessels():
    a = segments([(0, 0, 1, 55.0, 55.1)])
    b = segments([(0, 5, 6, 55.2, 55.3)]).with_columns(mmsi=pl.lit(219_000_002, pl.Int64))
    assert find_gaps(pl.concat([a, b]), 2.0).is_empty()


def coverage_for(cell_lat, hours, n):
    return pl.DataFrame({
        "cx": [140] * len(hours), "cy": [int(cell_lat / 0.1)] * len(hours),
        "hour": [h(x) for x in hours], "n": [n] * len(hours),
    }, schema={"cx": pl.Int32, "cy": pl.Int32, "hour": pl.Datetime("us", "UTC"),
               "n": pl.UInt32})


def test_coverage_uses_interior_hours_only():
    g = find_gaps(segments([(0, 0, 1.5, 55.0, 55.05), (1, 5.5, 6, 55.25, 55.3)]), 2.0)
    # gap 01:30 -> 05:30: interior hours are 02, 03, 04
    cov = pl.concat([coverage_for(55.05, [1, 2, 3, 4], 5),     # start cell busy
                     coverage_for(55.25, [2, 3], 5)])          # end cell: 2 of 3 hours
    r = add_coverage(g, cov, DP).row(0, named=True)
    assert r["coverage_start"] == pytest.approx(1.0)
    assert r["coverage_end"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("start_lat, end_lat, cov_n, port_lat, expected", [
    (55.05, 55.25, 5, None, "dark"),
    (55.05, 55.25, 1, None, "low_coverage"),              # receivers heard nobody
    (55.05, 57.95, 5, None, "position_jump"),             # 174 NM in 4 h = 43 kn
    (55.05, 55.25, 5, 55.05, "port_or_anchorage"),
])
def test_classify_gaps(start_lat, end_lat, cov_n, port_lat, expected):
    g = find_gaps(segments([(0, 0, 1.5, 55.0, start_lat), (1, 5.5, 6, end_lat, end_lat)]), 2.0)
    cov = pl.concat([coverage_for(start_lat, [2, 3, 4], cov_n),
                     coverage_for(end_lat, [2, 3, 4], cov_n)])
    zones = no_zones()
    if port_lat is not None:
        zones = Zones(pl.DataFrame({"lat": [port_lat], "lon": [14.05], "radius_km": [3.0]}),
                      zones.anchor_cells, 0.02)
    out = classify_gaps(add_coverage(g, cov, DP), zones, DP)
    assert out["category"].to_list() == [expected]


# ---------------------------------------------------------------- manoeuvres

def track(cogs, sogs, dt_s=60):
    """Dead-reckoned track from per-fix COG/SOG (one fix per dt_s)."""
    import math
    lat, lon, lats, lons = 55.5, 14.5, [], []
    for cog, sog in zip(cogs, sogs, strict=True):
        lats.append(lat)
        lons.append(lon)
        d = sog * KNOT_MS * dt_s
        lat += d * math.cos(math.radians(cog)) / M_PER_DEG_LAT
        lon += d * math.sin(math.radians(cog)) / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    n = len(cogs)
    return pl.DataFrame({
        "mmsi": [219_000_001] * n, "seg": [0] * n,
        "ts": [T0 + timedelta(seconds=i * dt_s) for i in range(n)],
        "lat": lats, "lon": lons, "sog": sogs, "cog": cogs,
    })


def test_straight_track_has_no_events():
    assert detect_maneuvers(track([45.0] * 40, [12.0] * 40), MP).is_empty()


def test_u_turn_detected_once():
    df = track([10.0] * 20 + [190.0] * 20, [12.0] * 40)
    ev = detect_maneuvers(df, MP)
    assert ev["event"].to_list() == ["u_turn"]
    assert ev["turn_deg"][0] == pytest.approx(180.0)


def test_u_turn_across_north_wraps():
    df = track([355.0] * 20 + [170.0] * 20, [12.0] * 40)   # 175 deg turn, not 185
    assert detect_maneuvers(df, MP)["event"].to_list() == ["u_turn"]


def test_abrupt_stop_detected():
    df = track([90.0] * 40, [12.0] * 20 + [1.0] * 20)
    assert detect_maneuvers(df, MP)["event"].to_list() == ["abrupt_stop"]


def test_no_comparison_across_segments():
    df = track([10.0] * 20 + [190.0] * 20, [12.0] * 40)
    df = df.with_columns(seg=pl.when(pl.int_range(pl.len()) >= 20).then(1).otherwise(0))
    assert detect_maneuvers(df, MP).is_empty()
