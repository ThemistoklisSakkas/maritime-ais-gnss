from datetime import timedelta

import polars as pl
import pytest

from maritime.ais.clean import with_jump_flag
from maritime.ais.trajectories import build_trajectories, segment, summarize_segments
from tests.test_clean import P, track


def prepared(df):
    return with_jump_flag(df.sort(["mmsi", "ts"]), P)


def test_continuous_track_is_one_segment():
    out = segment(prepared(track()), max_gap_s=1800)
    assert out["seg"].unique().to_list() == [0]
    assert out["dist_prev_m"][0] is None
    assert out["dist_prev_m"].null_count() == 1


def test_time_gap_starts_new_segment():
    df = track(n=10).with_columns(
        pl.when(pl.int_range(pl.len()) >= 5).then(pl.col("ts") + timedelta(hours=3))
        .otherwise(pl.col("ts")).alias("ts")
    )
    out = segment(prepared(df), max_gap_s=1800)
    assert out["seg"].to_list() == [0] * 5 + [1] * 5
    assert out["dist_prev_m"][5] is None  # no distance across the gap


def test_jump_starts_new_segment():
    df = track(n=10).with_columns(
        pl.when(pl.int_range(pl.len()) >= 5).then(pl.col("lon") + 0.1).otherwise(pl.col("lon"))
        .alias("lon")
    )
    out = segment(prepared(df), max_gap_s=1800)
    assert out["seg"].to_list() == [0] * 5 + [1] * 5


def test_segments_numbered_per_vessel():
    df = pl.concat([track(mmsi=219_000_001, n=4), track(mmsi=219_000_002, n=4, lat0=56.0)])
    out = segment(prepared(df), max_gap_s=1800)
    assert out.group_by("mmsi").agg(pl.col("seg").max())["seg"].to_list() == [0, 0]


def test_summary_distance_and_speed():
    df = track(n=361, dt_s=10, sog_kn=10.0)  # exactly 1 hour at 10 kn
    seg = summarize_segments(segment(prepared(df).with_columns(sog=pl.lit(10.0)), 1800))
    r = seg.row(0, named=True)
    assert r["duration_h"] == pytest.approx(1.0)
    assert r["length_nm"] == pytest.approx(10.0, rel=1e-3)
    assert r["mean_track_speed_kn"] == pytest.approx(10.0, rel=1e-3)


def test_build_trajectories_buckets(tmp_path):
    cols = {"sog": pl.lit(10.0, pl.Float32), "cog": pl.lit(0.0, pl.Float32),
            "heading": pl.lit(0.0, pl.Float32), "nav_status": pl.lit("Under way")}
    day1 = pl.concat([track(mmsi=219_000_001), track(mmsi=219_000_002, lat0=56.0)])
    day1.with_columns(**cols).write_parquet(tmp_path / "d1.parquet")
    stats = build_trajectories([tmp_path / "d1.parquet"], tmp_path / "traj", P,
                               max_gap_min=30, n_buckets=4)
    assert stats["n_vessels"] == 2
    assert stats["n_segments"] == 2
    assert stats["n_points"] == day1.height
    assert (tmp_path / "traj" / "segments.parquet").exists()
