from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from maritime.ais.eta import (
    ETAParams,
    arrivals_from_stays,
    build_samples,
    destination_matches,
    evaluate,
    find_stays,
    parse_crew_eta,
    time_split,
)

T0 = datetime(2025, 6, 3, tzinfo=UTC)
P = ETAParams(ship_types=("Cargo", "Tanker"), resample_min=1, min_stay_h=1.0, stop_sog_kn=1.0,
              min_stop_share=0.5, sample_every_min=30, max_horizon_h=48,
              state_tolerance_min=10, sog_floor_kn=1.0, train_share=0.7, zone_cell_deg=0.005,
              crew_plausible_max_h=336)
PORTS = pl.DataFrame({"port": ["Testport"], "locode": ["DK TST"], "lat": [56.0],
                      "lon": [11.0], "radius_km": [3.0]})
M_PER_DEG_LAT = 111_195.0
MMSI = 219_000_001


def approach_track(mmsi=MMSI, dist_nm=20.0, sog=10.0, stay_h=2.0, start=T0):
    """Due-north approach at constant speed to the port, then alongside (SOG 0)."""
    minutes = int(dist_nm / sog * 60)
    rows = []
    for i in range(minutes + 1):
        d_left_m = (dist_nm - sog * i / 60) * 1852
        lat = 56.0 - d_left_m / M_PER_DEG_LAT
        in_port = d_left_m <= 3000
        rows.append((mmsi, start + timedelta(minutes=i), lat, 11.0, sog, 0.0,
                     0 if in_port else -1, sog))
    t_end = rows[-1][1]
    for i in range(1, int(stay_h * 60) + 1):
        rows.append((mmsi, t_end + timedelta(minutes=i), 56.0, 11.0, 0.0, 0.0, 0, 0.0))
    return pl.DataFrame(rows, schema=["mmsi", "ts", "lat", "lon", "sog", "cog", "port",
                                      "sog_mean_1h"], orient="row")


def registry():
    return pl.DataFrame({"mmsi": [MMSI, MMSI + 1], "length": [150.0, 90.0],
                         "ship_type": ["Cargo", "Tanker"]})


# ---------------------------------------------------------------- arrivals

def test_stay_and_arrival_time():
    tr = approach_track()
    stays = find_stays(tr, P)
    assert stays.height == 1
    arr = arrivals_from_stays(stays, tr)
    entry = tr.filter(pl.col("port") == 0)["ts"].min()
    assert arr["arrival"].to_list() == [entry]
    assert arr["trip_start"].to_list() == [T0]


def test_passing_through_port_zone_is_not_an_arrival():
    tr = approach_track(stay_h=0).with_columns(sog=pl.lit(10.0))   # never stops
    assert find_stays(tr, P).is_empty()


def test_vessel_already_in_port_at_data_start_is_not_an_arrival():
    tr = approach_track()
    in_port_from_start = tr.filter(pl.col("port") == 0)
    stays = find_stays(in_port_from_start, P)
    assert arrivals_from_stays(stays, in_port_from_start).is_empty()


def test_trip_starts_after_previous_stay():
    first = approach_track()
    second = approach_track(start=first["ts"].max() + timedelta(minutes=1))
    tr = pl.concat([first, second])
    arr = arrivals_from_stays(find_stays(tr, P), tr)
    # the second "arrival" is preceded by the first stay, so its trip starts at that stay's end
    assert arr.height >= 1
    assert (arr["trip_start"] < arr["arrival"]).all()


# ---------------------------------------------------------------- samples

def test_samples_target_and_straight_line_baseline():
    tr = approach_track(dist_nm=20, sog=10)             # 2 h run-in
    arr = arrivals_from_stays(find_stays(tr, P), tr)
    s = build_samples(arr, tr, PORTS, registry(), P)
    assert s.height > 0
    assert (s["target_h"] > 0).all() and (s["target_h"] <= 2).all()
    # straight track, constant speed: distance/speed is (almost) exact, up to the port radius
    radius_h = 3000 / 1852 / 10
    err = (s["baseline_h"] - s["target_h"]).to_numpy()
    assert np.allclose(err, radius_h, atol=0.05)
    assert s["is_tanker"].to_list() == [0] * s.height


def test_samples_respect_max_horizon():
    tr = approach_track(dist_nm=60, sog=1.2)            # ~50 h run-in
    arr = arrivals_from_stays(find_stays(tr, P), tr)
    s = build_samples(arr, tr, PORTS, registry(), P)
    assert s["target_h"].max() <= 48


# ---------------------------------------------------------------- crew ETA

@pytest.mark.parametrize("eta, reported, expected", [
    ("04/06/2025 12:30:00", datetime(2025, 6, 3, tzinfo=UTC), datetime(2025, 6, 4, 12, 30)),
    ("06/06/2026 10:00:00", datetime(2025, 6, 5, tzinfo=UTC), datetime(2025, 6, 6, 10, 0)),
    ("02/01/2025 08:00:00", datetime(2025, 12, 30, tzinfo=UTC), datetime(2026, 1, 2, 8, 0)),
])
def test_parse_crew_eta_fixes_year(eta, reported, expected):
    df = pl.DataFrame({"eta": [eta], "reported": [reported]})
    out = df.select(parse_crew_eta(pl.col("eta"), pl.col("reported"))).item()
    assert out == expected.replace(tzinfo=UTC)


@pytest.mark.parametrize("dest, ok", [
    ("DK TST", True), ("DKTST", True), ("dk-tst", True),
    ("TESTPORT", True), ("TESTPORT ANCH.", True),
    ("NLRTM", False), ("FOR ORDERS", False), (None, False), ("DK", False), ("TST", False),
])
def test_destination_matching(dest, ok):
    assert destination_matches(dest, "DK TST", "Testport") is ok


# ---------------------------------------------------------------- split and metrics

def test_time_split_keeps_voyages_whole():
    s = pl.DataFrame({"arrival_id": [i for i in range(10) for _ in range(3)],
                      "arrival": [T0 + timedelta(hours=i) for i in range(10) for _ in range(3)]})
    train, test = time_split(s, 0.7)
    assert set(train["arrival_id"]) & set(test["arrival_id"]) == set()
    assert train["arrival"].max() < test["arrival"].min()


def test_evaluate_like_for_like_crew_subsets():
    test = pl.DataFrame({
        "target_h": [1.0, 2.0, 10.0, 30.0],
        "baseline_h": [2.0, 2.0, 12.0, 30.0],
        "model_h": [1.0, 3.0, 10.0, 31.0],
        "crew_h": [None, 2.5, -50.0, None],      # one sensible, one stale (already passed)
    })
    row = evaluate(test, 336).filter(pl.col("horizon") == "all").row(0, named=True)
    assert row["mae_baseline_h"] == pytest.approx((1 + 0 + 2 + 0) / 4)
    assert row["mae_model_h"] == pytest.approx((0 + 1 + 0 + 1) / 4)
    assert row["crew_n"] == 2
    assert row["crew_mae_crew_h"] == pytest.approx((0.5 + 60) / 2)
    assert row["crew_median_ae_crew_h"] == pytest.approx((0.5 + 60) / 2)
    assert row["plausible_n"] == 1
    assert row["plausible_mae_crew_h"] == pytest.approx(0.5)
    assert row["plausible_mae_model_h"] == pytest.approx(1.0)
