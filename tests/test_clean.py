from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from maritime.ais.clean import (
    CleaningParams,
    clean_day,
    drop_low_resolution,
    drop_stale_repeats,
    normalize_sentinels,
    remove_spikes,
    scan_raw_csv,
    stage_csv,
    valid_mmsi,
    valid_position,
    with_jump_flag,
)
from maritime.geo import EARTH_RADIUS_M, KNOT_MS

P = CleaningParams(
    mobile_types=("Class A", "Class B"),
    mmsi_min=201_000_000,
    mmsi_max=775_999_999,
    max_sog_kn=60.0,
    max_speed_kn=50.0,
    min_jump_m=200.0,
    spike_passes=3,
    stale_min_sog_kn=1.0,
    low_res_tol=1e-3,
)
T0 = datetime(2025, 6, 2, tzinfo=UTC)
M_PER_DEG_LAT = EARTH_RADIUS_M * 3.141592653589793 / 180


def track(mmsi=219_000_001, n=20, dt_s=10, sog_kn=10.0, lat0=55.0123, lon0=11.0456):
    """Straight northbound track at constant speed."""
    step = sog_kn * KNOT_MS * dt_s / M_PER_DEG_LAT
    return pl.DataFrame({
        "mmsi": [mmsi] * n,
        "ts": [T0 + timedelta(seconds=i * dt_s) for i in range(n)],
        "lat": [lat0 + i * step for i in range(n)],
        "lon": [lon0] * n,
    })


# ---------------------------------------------------------------- field rules

@pytest.mark.parametrize("mmsi, ok", [
    (219_461_000, True),    # Danish ship
    (775_999_999, True),    # upper MID bound
    (2_190_064, False),     # base station (00 + MID)
    (111_219_500, False),   # SAR aircraft
    (970_123_456, False),   # AIS-SART
    (992_191_234, False),   # aid to navigation
    (123_456_789, False),   # default/garbage
])
def test_valid_mmsi(mmsi, ok):
    df = pl.DataFrame({"mmsi": [mmsi]})
    assert df.select(valid_mmsi(P)).item() is ok


@pytest.mark.parametrize("lat, lon, ok", [
    (55.5, 11.0, True),
    (91.0, 11.0, False),   # AIS latitude not available
    (55.5, 181.0, False),  # AIS longitude not available
    (0.0, 0.0, False),     # null island
])
def test_valid_position(lat, lon, ok):
    df = pl.DataFrame({"ts": [T0], "lat": [lat], "lon": [lon]})
    assert df.select(valid_position()).item() is ok


def test_sentinels_become_null():
    df = pl.DataFrame({"sog": [102.3, 12.0], "cog": [360.0, 45.0],
                       "heading": [511.0, 44.0], "rot": [-720.0, 3.0]})
    out = df.with_columns(normalize_sentinels(P))
    assert out.row(0) == (None, None, None, None)
    assert out.row(1) == (12.0, 45.0, 44.0, 3.0)


# ---------------------------------------------------------------- report artefacts

@pytest.mark.parametrize("lat, lon, low_res", [
    (56.848333, 11.815, True),        # 56 deg 50.9', 11 deg 48.9': msg-27 style
    (56.848747, 11.816300, False),    # normal 1/10000' report
    (56.848333, 11.816300, False),    # only one axis on the grid: chance, keep it
])
def test_low_resolution_detection(lat, lon, low_res):
    df = pl.DataFrame({"lat": [lat], "lon": [lon]})
    _, n = drop_low_resolution(df, P)
    assert n == int(low_res)


def test_stale_repeat_dropped_only_when_moving():
    df = pl.DataFrame({
        "mmsi": [1, 1, 1, 2, 2],
        "ts": [T0 + timedelta(seconds=s) for s in (0, 30, 40, 0, 30)],
        "lat": [55.1, 55.1, 55.2, 56.1, 56.1],
        "lon": [11.1, 11.1, 11.1, 12.1, 12.1],
        "sog": [9.7, 9.7, 9.7, 0.0, 0.0],   # vessel 1 under way, vessel 2 moored
    })
    out, n = drop_stale_repeats(df, P)
    assert n == 1
    assert out.filter(pl.col("mmsi") == 2).height == 2  # moored repeats kept


def test_stale_repeat_removal_prevents_false_jump():
    # Ship at 40 kn (~20.6 m/s). True position at time t is lat0 + v*t.
    v = 40.0 * KNOT_MS / M_PER_DEG_LAT                     # deg/s
    t = [0, 10, 20, 30, 40, 50, 52, 62]                    # fix at 50 s is stale, next at 52 s
    lat = [55.0123 + v * s for s in t]
    lat[5] = lat[4]                                        # 50 s fix repeats the 40 s position
    df = pl.DataFrame({
        "mmsi": [219_000_001] * len(t),
        "ts": [T0 + timedelta(seconds=s) for s in t],
        "lat": lat, "lon": [11.0456] * len(t), "sog": [40.0] * len(t),
    })
    # stale(50 s) -> 52 s looks like ~250 m in 2 s: fake jump
    assert with_jump_flag(df, P)["jump"].any()
    cleaned, n = drop_stale_repeats(df, P)
    assert n == 1
    assert not with_jump_flag(cleaned, P)["jump"].any()


# ---------------------------------------------------------------- kinematics

def test_clean_track_has_no_jumps_or_spikes():
    out, n = remove_spikes(track(), P)
    assert n == 0
    assert not out["jump"].any()


def test_isolated_spike_is_removed():
    df = track()
    df = df.with_columns(  # push fix 10 five km east: a classic multipath/garbage fix
        pl.when(pl.int_range(pl.len()) == 10).then(pl.col("lon") + 0.08).otherwise(pl.col("lon"))
        .alias("lon")
    )
    out, n = remove_spikes(df, P)
    assert n == 1
    assert out.height == df.height - 1
    assert not out["jump"].any()


def test_relocation_is_flagged_not_removed():
    df = track()
    df = df.with_columns(  # from fix 10 on, the whole track sits 5 km east (spoofing-like)
        pl.when(pl.int_range(pl.len()) >= 10).then(pl.col("lon") + 0.08).otherwise(pl.col("lon"))
        .alias("lon")
    )
    out, n = remove_spikes(df, P)
    assert n == 0
    assert out.height == df.height
    assert out["jump"].sum() == 1
    assert out["jump"].arg_max() == 10


def test_gnss_noise_at_short_interval_is_not_a_jump():
    df = track(n=5, dt_s=1, sog_kn=0.0)
    df = df.with_columns(pl.Series("lat", [55.0, 55.00027, 55.0, 55.00027, 55.0]))  # 30 m jitter
    assert not with_jump_flag(df, P)["jump"].any()


def test_no_jump_across_different_vessels():
    a = track(mmsi=219_000_001)
    b = track(mmsi=219_000_002, lat0=57.0)  # far away from a, same timestamps
    df = pl.concat([a, b]).sort(["mmsi", "ts"])
    assert not with_jump_flag(df, P)["jump"].any()


# ---------------------------------------------------------------- end to end

HEADER = ("# Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,ROT,SOG,COG,"
          "Heading,IMO,Callsign,Name,Ship type,Cargo type,Width,Length,"
          "Type of position fixing device,Draught,Destination,ETA,Data source type,A,B,C,D")


def row(ts, mobile, mmsi, lat, lon, sog="10.0", name="", dest="Unknown"):
    return (f"{ts},{mobile},{mmsi},{lat},{lon},Under way using engine,0.0,{sog},10.0,10,"
            f"Unknown,Unknown,{name},Cargo,,20,150,GPS,7.5,{dest},,AIS,,,,")


@pytest.fixture
def raw_csv(tmp_path):
    lines = [
        HEADER,
        row("02/06/2025 00:00:00", "Class A", 219000001, 55.0123, 11.0456),
        row("02/06/2025 00:00:00", "Class A", 219000001, 55.0123, 11.0456),          # duplicate
        row("02/06/2025 00:00:10", "Class A", 219000001, 55.0128, 11.0456, name="TEST SHIP",
            dest="AARHUS"),
        row("02/06/2025 00:00:20", "Class A", 219000001, 55.0133, 11.0456, sog="102.3"),
        row("02/06/2025 00:00:00", "Base Station", 2190064, 56.7, 11.5),       # not a vessel
        row("02/06/2025 00:00:00", "Class A", 123456789, 55.0123, 11.0456),          # bad MMSI
        row("02/06/2025 00:00:00", "Class A", 219000002, 91.0, 181.0),         # no position
        row("02/06/2025 00:00:00", "Class B", 219000003, 40.0, 0.5),           # outside AOI
    ]
    p = tmp_path / "aisdk-2025-06-02.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_scan_raw_csv_types_and_null_tokens(raw_csv):
    df = scan_raw_csv(raw_csv).collect()
    assert df["ts"].dtype == pl.Datetime("us", "UTC")
    assert df["ts"][2] == T0 + timedelta(seconds=10)
    assert df["name"][0] is None          # empty -> null
    assert df["destination"][0] is None   # "Unknown" -> null
    assert df["destination"][2] == "AARHUS"


def test_clean_day_funnel(raw_csv, tmp_path):
    stage = stage_csv(raw_csv, tmp_path / "stage.parquet")
    stats = clean_day(stage, tmp_path / "pos.parquet", tmp_path / "static.parquet",
                      P, bbox=(3.0, 53.5, 17.0, 58.5))
    assert stats["n_raw"] == 8
    assert stats["n_not_vessel"] == 1
    assert stats["n_bad_mmsi"] == 1
    assert stats["n_bad_position"] == 1
    assert stats["n_outside_aoi"] == 1
    assert stats["n_duplicates"] == 1
    assert stats["n_clean"] == 3
    assert stats["n_vessels"] == 1

    pos = pl.read_parquet(tmp_path / "pos.parquet")
    assert pos["sog"].null_count() == 1   # 102.3 sentinel
    static = pl.read_parquet(tmp_path / "static.parquet")
    assert "AARHUS" in static["destination"].to_list()
