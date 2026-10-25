import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import shapely

from maritime.ais.gnss_anomaly import (
    CircleParams,
    InterferenceParams,
    cell_hour_stats,
    convergence,
    detect_circles,
    distance_to_land_m,
    flag_cells,
    inland_area,
    landed_after_jump,
    points_on_land,
)
from maritime.geo import KNOT_MS, from_metric, to_metric

CRS = "EPSG:25832"
T0 = datetime(2025, 6, 5, 14, tzinfo=UTC)
P = InterferenceParams(inland_buffer_m=1000, cell_deg=0.5, min_affected_vessels=3,
                       min_affected_share=0.05,
                       convergence_cell_deg=0.05, convergence_min_vessels=3)
CP = CircleParams(window_min=30, min_turn_deg=360, min_sog_kn=3, max_dt_s=120, merge_min=60,
                  expected_types=("Fishing", "SAR"))


# ---------------------------------------------------------------- land

@pytest.fixture
def square_island():
    """A 10 km x 10 km island centred at 55.5N 12.5E (built in metres)."""
    cx, cy = to_metric(12.5, 55.5, CRS)
    return shapely.box(cx - 5000, cy - 5000, cx + 5000, cy + 5000), (cx, cy)


def test_points_on_land_respects_inland_buffer(square_island):
    land, (cx, cy) = square_island
    inland = inland_area(land, 1000)
    xs = np.array([cx, cx + 4500, cx + 7000])        # centre, 500 m inside edge, 2 km at sea
    lon, lat = from_metric(xs, np.full(3, cy), CRS)
    assert points_on_land(lon, lat, inland, CRS).tolist() == [True, False, False]


def test_distance_to_land(square_island):
    land, (cx, cy) = square_island
    lon, lat = from_metric(np.array([cx + 7000.0]), np.array([cy]), CRS)
    assert distance_to_land_m(lon, lat, land, CRS)[0] == pytest.approx(2000.0, abs=1.0)


# ---------------------------------------------------------------- cell statistics

def fixes(mmsis, jumps, lat=55.55, lon=15.05, ts=T0):
    n = len(mmsis)
    return pl.DataFrame({"mmsi": mmsis, "ts": [ts] * n, "lat": [lat] * n, "lon": [lon] * n,
                         "jump": jumps})


def test_cell_hour_counts_distinct_vessels():
    df = fixes([1, 1, 2, 3], [True, True, True, False])
    stats, _ = cell_hour_stats(df, np.array([False, False, False, True]), P)
    r = stats.row(0, named=True)
    assert (r["n"], r["jump_vessels"], r["land_vessels"], r["affected"]) == (3, 2, 1, 3)


def test_vessel_with_jump_and_land_counted_once():
    df = fixes([1, 1], [True, False])
    stats, _ = cell_hour_stats(df, np.array([False, True]), P)
    assert stats.row(0, named=True)["affected"] == 1


def test_land_counts_only_after_a_jump():
    df = pl.DataFrame({
        "mmsi": [1, 1, 1, 2, 2],
        "seg": [0, 1, 1, 0, 0],
        "jump": [False, True, False, False, False],   # vessel 1 jumps into segment 1
    })
    on_land = np.array([False, True, True, True, True])  # vessel 2: fjord the map closes
    assert landed_after_jump(df, on_land).tolist() == [False, True, True, False, False]


def stats_row(n, jumps, land, affected, cx=30, cy=111, hour=T0):
    return {"cx": cx, "cy": cy, "hour": hour, "n": n, "jump_vessels": jumps,
            "land_vessels": land, "affected": affected}


def as_stats(rows):
    return pl.DataFrame(rows, schema={"cx": pl.Int32, "cy": pl.Int32,
                                      "hour": pl.Datetime("us", "UTC"), "n": pl.UInt32,
                                      "jump_vessels": pl.UInt32, "land_vessels": pl.UInt32,
                                      "affected": pl.UInt32})


@pytest.mark.parametrize("n, affected, flagged", [
    (40, 4, True),      # 4 of 40 vessels: 10 %
    (40, 2, False),     # only 2 vessels
    (200, 4, False),    # 4 of 200 = 2 % < 5 %
])
def test_flag_cells(n, affected, flagged):
    out = flag_cells(as_stats([stats_row(n, affected, 0, affected)]), P)
    assert out["flagged"].to_list() == [flagged]


def test_flag_cells_sums_buckets():
    rows = [stats_row(20, 2, 0, 2), stats_row(20, 2, 0, 2)]   # same cell-hour, two buckets
    out = flag_cells(as_stats(rows), P)
    assert out.height == 1
    assert out.row(0, named=True)["affected"] == 4
    assert out["flagged"].to_list() == [True]


def test_convergence_vs_scatter():
    together = fixes([1, 2, 3], [True] * 3, lat=55.61, lon=15.01)
    scattered = pl.concat([fixes([4], [True], lat=55.2, lon=14.6),
                           fixes([5], [True], lat=55.9, lon=15.9),
                           fixes([6], [True], lat=56.3, lon=16.4)])
    _, dest_t = cell_hour_stats(together, np.zeros(3, bool), P)
    _, dest_s = cell_hour_stats(scattered, np.zeros(3, bool), P)
    assert convergence(dest_t, P).height == 1
    assert convergence(dest_s, P).is_empty()


# ---------------------------------------------------------------- circles

def turning_track(rate_deg_min, minutes=40, sog=8.0, mmsi=219_000_001):
    lat, lon, cog = 55.5, 14.5, 0.0
    rows = []
    for i in range(minutes + 1):
        rows.append((mmsi, 0, T0 + timedelta(minutes=i), lat, lon, sog, cog % 360))
        d = sog * KNOT_MS * 60
        lat += d * math.cos(math.radians(cog)) / 111_195
        lon += d * math.sin(math.radians(cog)) / (111_195 * math.cos(math.radians(lat)))
        cog += rate_deg_min
    return pl.DataFrame(rows, schema=["mmsi", "seg", "ts", "lat", "lon", "sog", "cog"],
                        orient="row")


def test_full_circle_detected_once():
    ev = detect_circles(turning_track(15), CP)      # 15 deg/min: 360 deg in 24 min
    assert ev.height == 1
    assert ev["turn"][0] >= 360


def test_counter_clockwise_circle_detected():
    assert detect_circles(turning_track(-15), CP).height == 1


def test_gentle_turn_and_straight_line_ignored():
    assert detect_circles(turning_track(5), CP).is_empty()   # 150 deg in 30 min
    assert detect_circles(turning_track(0), CP).is_empty()


def test_slow_circle_ignored():
    assert detect_circles(turning_track(15, sog=2.0), CP).is_empty()
