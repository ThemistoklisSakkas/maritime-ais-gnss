"""Per-vessel trajectories: split each MMSI's cleaned fixes into continuous segments.

A new segment starts when
  - the time since the previous fix exceeds `max_gap_min` (reception loss or AIS off), or
  - the previous fix -> this fix move is physically impossible (`jump`, see clean.py).

Segments are the unit later phases work on: A2 inspects what happens *between*
segments (dark periods), A3/A4 work *within* them.

Memory: all days are needed at once (voyages cross midnight), so vessels are
processed in `n_buckets` groups by `mmsi % n_buckets`. Each bucket holds every
fix of ~1/n of the fleet, which keeps peak RAM bounded on an 8 GB machine.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from maritime.ais.clean import CleaningParams, with_jump_flag
from maritime.geo import haversine_expr

TRAJ_COLS = ["ts", "mmsi", "lat", "lon", "sog", "cog", "heading", "nav_status"]


def segment(df: pl.DataFrame, max_gap_s: float) -> pl.DataFrame:
    """Add seg (per-MMSI segment number), dist_prev_m and dt_prev_s.

    Requires df sorted by (mmsi, ts) and a boolean `jump` column.
    dist/dt are null at the first fix of every segment.
    """
    c = pl.col
    same_prev = c("mmsi") == c("mmsi").shift(1)
    dt = (c("ts") - c("ts").shift(1)).dt.total_seconds()
    new_seg = (~same_prev | (dt > max_gap_s) | c("jump")).fill_null(True)

    df = df.with_columns(
        (new_seg.cast(pl.Int32).cum_sum().over("mmsi") - 1).alias("seg"),
        new_seg.alias("_new"),
        dt.alias("_dt"),
        haversine_expr(c("lat").shift(1), c("lon").shift(1), c("lat"), c("lon")).alias("_d"),
    )
    return df.with_columns(
        pl.when(~c("_new")).then(c("_d")).alias("dist_prev_m"),
        pl.when(~c("_new")).then(c("_dt")).alias("dt_prev_s"),
    ).drop("_new", "_dt", "_d")


def summarize_segments(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (mmsi, seg) with start/end, duration, distance and speeds."""
    c = pl.col
    return (
        df.group_by(["mmsi", "seg"])
        .agg(
            start=c("ts").first(),
            end=c("ts").last(),
            n_points=pl.len(),
            start_lat=c("lat").first(), start_lon=c("lon").first(),
            end_lat=c("lat").last(), end_lon=c("lon").last(),
            length_nm=c("dist_prev_m").sum() / 1852.0,
            mean_sog_kn=c("sog").mean(),
            starts_with_jump=c("jump").first(),
        )
        .with_columns(duration_h=(c("end") - c("start")).dt.total_seconds() / 3600.0)
        .with_columns(
            mean_track_speed_kn=pl.when(c("duration_h") > 0)
            .then(c("length_nm") / c("duration_h"))
        )
        .sort(["mmsi", "seg"])
    )


def build_trajectories(clean_paths: list[Path], out_dir: Path, p: CleaningParams,
                       max_gap_min: float, n_buckets: int) -> dict:
    """Segment all days' positions, bucket by MMSI. Writes points_bXX.parquet + segments.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    max_gap_s = max_gap_min * 60.0
    summaries = []
    n_points = 0

    for b in range(n_buckets):
        df = (
            pl.scan_parquet(clean_paths)
            .filter(pl.col("mmsi") % n_buckets == b)
            .select(TRAJ_COLS)
            .collect(engine="streaming")
            .sort(["mmsi", "ts"])
        )
        if df.is_empty():
            continue
        # Recompute jumps on the multi-day series so midnight crossings are checked too
        df = segment(with_jump_flag(df, p), max_gap_s)
        df.write_parquet(out_dir / f"points_b{b:02d}.parquet")
        summaries.append(summarize_segments(df))
        n_points += df.height

    seg = pl.concat(summaries).sort(["mmsi", "seg"])
    seg.write_parquet(out_dir / "segments.parquet")

    return {
        "n_points": n_points,
        "n_vessels": seg["mmsi"].n_unique(),
        "n_segments": seg.height,
        "n_segments_from_jump": int(seg["starts_with_jump"].sum()),
        "median_segment_duration_h": float(seg["duration_h"].median()),
        "total_track_nm": float(seg["length_nm"].sum()),
    }
