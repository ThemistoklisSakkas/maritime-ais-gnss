"""A2 - Dark activity: AIS transmission gaps and abrupt manoeuvres.

A gap is "dark" only if we can rule out the boring explanations:

  1. Routine area  - it starts or ends in a port/anchorage (AIS off alongside is normal).
  2. No coverage   - shore receivers could not hear *anyone* there at the time. We check
                     this from the data itself: during every hour of the gap, were at least
                     `coverage_min_vessels` OTHER vessels received in the same grid cell?
  3. Not a silence - covering the gap would need > `max_gap_speed_kn`, so it is a position
                     jump (spoofing/MMSI clash, handled in A4), not a vessel going quiet.

Abrupt manoeuvres (U-turns, sudden stops at sea) are separate events: on their own they are
normal seamanship, but next to a dark gap or an encounter they become context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import polars as pl

from maritime.geo import angle_diff_expr, haversine_expr
from maritime.zones import Zones, cell_exprs


@dataclass(frozen=True)
class DarkParams:
    min_gap_h: float
    max_gap_speed_kn: float
    coverage_cell_deg: float
    coverage_min_vessels: int
    coverage_min_fraction: float
    min_move_nm: float

    @classmethod
    def from_dict(cls, d: dict) -> DarkParams:
        return cls(float(d["min_gap_h"]), float(d["max_gap_speed_kn"]),
                   float(d["coverage_cell_deg"]), int(d["coverage_min_vessels"]),
                   float(d["coverage_min_fraction"]), float(d["min_move_nm"]))


@dataclass(frozen=True)
class ManeuverParams:
    window_min: float
    min_turn_deg: float
    turn_min_sog_kn: float
    stop_from_kn: float
    stop_to_kn: float
    merge_min: float

    @classmethod
    def from_dict(cls, d: dict) -> ManeuverParams:
        return cls(*(float(d[k]) for k in ("window_min", "min_turn_deg", "turn_min_sog_kn",
                                           "stop_from_kn", "stop_to_kn", "merge_min")))


# ---------------------------------------------------------------- gaps

def find_gaps(segments: pl.DataFrame, min_gap_h: float) -> pl.DataFrame:
    """Silences between consecutive segments of the same MMSI lasting >= min_gap_h.

    Segments are already split on > 30 min silence (A1), so every inter-segment boundary
    is a candidate; jump-induced splits have ~0 duration and drop out here.
    """
    c = pl.col
    nxt = lambda col: c(col).shift(-1).over("mmsi")  # noqa: E731
    gaps = (
        segments.sort(["mmsi", "seg"])
        .select(
            "mmsi",
            c("end").alias("gap_start"),
            nxt("start").alias("gap_end"),
            c("end_lat").alias("start_lat"), c("end_lon").alias("start_lon"),
            nxt("start_lat").alias("end_lat"), nxt("start_lon").alias("end_lon"),
            c("end_sog_kn").alias("sog_before_kn"), nxt("start_sog_kn").alias("sog_after_kn"),
        )
        .drop_nulls("gap_end")
        .with_columns(gap_h=(c("gap_end") - c("gap_start")).dt.total_seconds() / 3600.0)
        .filter(c("gap_h") >= min_gap_h)
    )
    return gaps.with_columns(
        gap_dist_nm=haversine_expr(c("start_lat"), c("start_lon"),
                                   c("end_lat"), c("end_lon")) / 1852.0,
    ).with_columns(implied_speed_kn=c("gap_dist_nm") / c("gap_h"))


def coverage_table(points_paths: list[Path], cell_deg: float) -> pl.DataFrame:
    """Distinct vessels received per (cell, hour). Our own empirical receiver-coverage map."""
    parts = []
    for p in points_paths:  # MMSI buckets: distinct counts are additive across files
        parts.append(
            pl.scan_parquet(p)
            .with_columns(*cell_exprs(cell_deg), hour=pl.col("ts").dt.truncate("1h"))
            .group_by(["cx", "cy", "hour"])
            .agg(n=pl.col("mmsi").n_unique())
            .collect()
        )
    return pl.concat(parts).group_by(["cx", "cy", "hour"]).agg(pl.col("n").sum())


def add_coverage(gaps: pl.DataFrame, coverage: pl.DataFrame, p: DarkParams) -> pl.DataFrame:
    """Fraction of the gap's interior hours in which the start/end cell was 'covered'.

    Interior hours only (strictly after the start hour, before the end hour), so the
    vessel's own last/first reports never count as coverage.
    """
    g = gaps.with_row_index("gid")
    out = g
    for end in ("start", "end"):
        hours = (
            g.select(
                "gid",
                *cell_exprs(p.coverage_cell_deg, f"{end}_lat", f"{end}_lon"),
                hour=pl.datetime_ranges(
                    pl.col("gap_start").dt.truncate("1h") + timedelta(hours=1),
                    pl.col("gap_end").dt.truncate("1h") - timedelta(hours=1),
                    "1h",
                ),
            )
            .explode("hour")
            .drop_nulls("hour")
            .join(coverage, on=["cx", "cy", "hour"], how="left")
            .group_by("gid")
            .agg(((pl.col("n").fill_null(0) >= p.coverage_min_vessels).mean())
                 .alias(f"coverage_{end}"))
        )
        out = out.join(hours, on="gid", how="left")
    return out.drop("gid")


def classify_gaps(gaps: pl.DataFrame, zones: Zones, p: DarkParams) -> pl.DataFrame:
    """Label each gap with the first explanation that applies, else 'dark'."""
    g = gaps.with_columns(
        start_in_zone=pl.Series(zones.mask(gaps, "start_lat", "start_lon")),
        end_in_zone=pl.Series(zones.mask(gaps, "end_lat", "end_lon")),
    )
    c = pl.col
    covered = ((c("coverage_start").fill_null(0) >= p.coverage_min_fraction)
               & (c("coverage_end").fill_null(0) >= p.coverage_min_fraction))
    return g.with_columns(
        category=pl.when(c("start_in_zone") | c("end_in_zone")).then(pl.lit("port_or_anchorage"))
        .when(c("implied_speed_kn") > p.max_gap_speed_kn).then(pl.lit("position_jump"))
        .when(~covered).then(pl.lit("low_coverage"))
        .otherwise(pl.lit("dark")),
        # Reappearing far from where it fell silent = it transited without AIS.
        # Staying put is more often a small vessel in a marina WPI does not list.
        moved_while_dark=c("gap_dist_nm") >= p.min_move_nm,
    )


# ---------------------------------------------------------------- manoeuvres

def detect_maneuvers(df: pl.DataFrame, p: ManeuverParams) -> pl.DataFrame:
    """U-turns and abrupt stops, comparing each fix with the state `window_min` earlier.

    Uses an as-of join (last fix at or before t - window) within the same segment, so a
    comparison never spans a gap or a jump. Consecutive flagged fixes are merged into one
    event per vessel and type.
    """
    c = pl.col
    window = timedelta(minutes=p.window_min)
    left = df.select("mmsi", "seg", "ts", "lat", "lon", "sog", "cog") \
             .with_columns(ts_ref=c("ts") - window).sort("ts_ref")
    right = df.select("mmsi", "seg", c("ts").alias("ts_then"), c("sog").alias("sog_then"),
                      c("cog").alias("cog_then")).sort("ts_then")
    # Both sides are globally sorted on time, hence sorted within every (mmsi, seg) group;
    # polars cannot verify that itself when `by` is given, so we tell it.
    j = left.join_asof(right, left_on="ts_ref", right_on="ts_then", by=["mmsi", "seg"],
                       strategy="backward", check_sortedness=False)
    # the earlier fix must really be ~window ago, not hours ago after a sparse stretch
    j = j.filter((c("ts") - c("ts_then")) <= window * 1.5)

    turn = angle_diff_expr(c("cog"), c("cog_then")).abs()
    flagged = j.with_columns(
        event=pl.when((turn >= p.min_turn_deg) & (c("sog") >= p.turn_min_sog_kn)
                      & (c("sog_then") >= p.turn_min_sog_kn)).then(pl.lit("u_turn"))
        .when((c("sog_then") >= p.stop_from_kn) & (c("sog") <= p.stop_to_kn))
        .then(pl.lit("abrupt_stop")),
        turn_deg=turn,
    ).drop_nulls("event").sort(["mmsi", "event", "ts"])

    if flagged.is_empty():
        return flagged.select("mmsi", "event", "ts", "lat", "lon", "sog_then", "sog",
                              "cog_then", "cog", "turn_deg")
    merge = timedelta(minutes=p.merge_min)
    new_event = ((c("mmsi") != c("mmsi").shift(1)) | (c("event") != c("event").shift(1))
                 | ((c("ts") - c("ts").shift(1)) > merge)).fill_null(True)
    return (
        flagged.with_columns(eid=new_event.cum_sum())
        .group_by("eid", maintain_order=True)
        .agg(c("mmsi", "event", "ts", "lat", "lon", "sog_then", "sog", "cog_then", "cog",
               "turn_deg").first(), n_fixes=pl.len())
        .drop("eid")
    )
