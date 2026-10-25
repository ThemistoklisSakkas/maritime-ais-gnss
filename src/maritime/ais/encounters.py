"""A3 - Ship-to-ship (STS) encounter detection.

Rule: two vessels < `max_dist_m` apart, both slower than `max_sog_kn`, for at least
`min_duration_h`, outside ports and anchorages.

Algorithm (avoids the O(n^2) all-pairs comparison):
  1. Keep only slow fixes outside routine zones, and drop vessel types that raft up
     legitimately (sailing / pleasure craft).
  2. Snap every vessel to one mean position per `bin_min` time bin.
  3. Per bin, project to metres (UTM 32N) and let a KD-tree return every pair closer than
     `max_dist_m`: O(n log n) per bin instead of O(n^2).
  4. Chain a pair's consecutive bins into episodes, tolerating `max_missing_bins` empty
     bins (AIS from anchored ships arrives only every ~3 min, and receptions drop).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from maritime.geo import to_metric
from maritime.zones import Zones


@dataclass(frozen=True)
class STSParams:
    bin_min: int
    max_sog_kn: float
    max_dist_m: float
    min_duration_h: float
    max_missing_bins: int
    zone_cell_deg: float
    exclude_ship_types: tuple[str, ...]
    service_ship_types: tuple[str, ...]
    dark_window_h: float

    @classmethod
    def from_dict(cls, d: dict) -> STSParams:
        return cls(int(d["bin_min"]), float(d["max_sog_kn"]), float(d["max_dist_m"]),
                   float(d["min_duration_h"]), int(d["max_missing_bins"]),
                   float(d["zone_cell_deg"]), tuple(d["exclude_ship_types"]),
                   tuple(d["service_ship_types"]), float(d["dark_window_h"]))


# ---------------------------------------------------------------- 1-2: candidates

def slow_positions(points: pl.DataFrame, p: STSParams) -> pl.DataFrame:
    """One mean position per (mmsi, time bin) for bins where the vessel was slow."""
    return (
        points.filter(pl.col("sog") < p.max_sog_kn)
        .with_columns(bin=pl.col("ts").dt.truncate(f"{p.bin_min}m"))
        .group_by(["mmsi", "bin"])
        .agg(pl.col("lat").mean(), pl.col("lon").mean(), pl.col("sog").mean())
    )


def drop_routine(cand: pl.DataFrame, zones: Zones, registry: pl.DataFrame,
                 p: STSParams) -> tuple[pl.DataFrame, dict]:
    """Remove candidates in ports/anchorages and of excluded ship types. Returns counts."""
    in_zone = zones.mask_gridded(cand, p.zone_cell_deg)
    n0 = cand.height
    cand = cand.filter(~pl.Series(in_zone))
    n_zone = n0 - cand.height
    excluded = registry.filter(pl.col("ship_type").is_in(list(p.exclude_ship_types)))["mmsi"]
    cand = cand.filter(~pl.col("mmsi").is_in(excluded.implode()))
    return cand, {"slow_vessel_bins": n0, "removed_in_zone": n_zone,
                  "removed_by_type": n0 - n_zone - cand.height, "candidate_bins": cand.height}


# ---------------------------------------------------------------- 3: spatial pairing

def pairs_within(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """Index pairs (i < j) with Euclidean distance <= r, via a KD-tree. Shape (k, 2)."""
    if x.size < 2:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(np.column_stack([x, y]))
    return tree.query_pairs(r, output_type="ndarray")


def close_pairs(cand: pl.DataFrame, p: STSParams, crs: str) -> pl.DataFrame:
    """All (mmsi_a < mmsi_b, bin) with the two vessels within max_dist_m in that bin."""
    x, y = to_metric(cand["lon"].to_numpy(), cand["lat"].to_numpy(), crs)
    cand = cand.with_columns(x=pl.Series(x), y=pl.Series(y)).sort("bin")
    mmsi = cand["mmsi"].to_numpy()
    lat, lon = cand["lat"].to_numpy(), cand["lon"].to_numpy()
    xs, ys = cand["x"].to_numpy(), cand["y"].to_numpy()

    out = []
    bins = cand["bin"]
    # row ranges of each bin in the time-sorted frame
    starts = np.flatnonzero(np.r_[True, bins.to_numpy()[1:] != bins.to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(cand)]
    for s, e in zip(starts, ends, strict=True):
        ij = pairs_within(xs[s:e], ys[s:e], p.max_dist_m)
        if ij.size == 0:
            continue
        i, j = ij[:, 0] + s, ij[:, 1] + s
        swap = mmsi[i] > mmsi[j]                  # canonical order: mmsi_a < mmsi_b
        a = np.where(swap, j, i)
        b = np.where(swap, i, j)
        out.append(pl.DataFrame({
            "mmsi_a": mmsi[a], "mmsi_b": mmsi[b],
            "bin": bins.gather(np.full(len(a), s)),
            "dist_m": np.hypot(xs[a] - xs[b], ys[a] - ys[b]),
            "lat": (lat[a] + lat[b]) / 2, "lon": (lon[a] + lon[b]) / 2,
        }))
    if not out:
        return pl.DataFrame(schema={"mmsi_a": pl.Int64, "mmsi_b": pl.Int64,
                                    "bin": bins.dtype, "dist_m": pl.Float64,
                                    "lat": pl.Float64, "lon": pl.Float64})
    return pl.concat(out)


# ---------------------------------------------------------------- 4: episodes

def build_episodes(pairs: pl.DataFrame, p: STSParams) -> pl.DataFrame:
    """Chain each pair's bins into episodes and keep those lasting >= min_duration_h."""
    c = pl.col
    bin_len = timedelta(minutes=p.bin_min)
    max_step = bin_len * (p.max_missing_bins + 1)
    new_ep = ((c("mmsi_a") != c("mmsi_a").shift(1)) | (c("mmsi_b") != c("mmsi_b").shift(1))
              | ((c("bin") - c("bin").shift(1)) > max_step)).fill_null(True)
    return (
        pairs.sort(["mmsi_a", "mmsi_b", "bin"])
        .with_columns(ep=new_ep.cum_sum())
        .group_by("ep")
        .agg(
            c("mmsi_a").first(), c("mmsi_b").first(),
            start=c("bin").min(), end=c("bin").max() + bin_len,
            n_bins=pl.len(),
            mean_dist_m=c("dist_m").mean(), min_dist_m=c("dist_m").min(),
            lat=c("lat").mean(), lon=c("lon").mean(),
        )
        .with_columns(duration_h=(c("end") - c("start")).dt.total_seconds() / 3600.0)
        .filter(c("duration_h") >= p.min_duration_h)
        .drop("ep")
        .sort("start")
    )


def classify_pairs(ep: pl.DataFrame, registry: pl.DataFrame, focus: list[str],
                   p: STSParams) -> pl.DataFrame:
    """Attach names/types and a pair class used for headline numbers."""
    reg = registry.select("mmsi", "name", "ship_type", "length")
    ep = (ep.join(reg.rename(lambda n: f"{n}_a"), on="mmsi_a", how="left")
            .join(reg.rename(lambda n: f"{n}_b"), on="mmsi_b", how="left"))
    c = pl.col
    ta, tb = c("ship_type_a"), c("ship_type_b")
    focus_a, focus_b = ta.is_in(focus).fill_null(False), tb.is_in(focus).fill_null(False)
    service = (ta.is_in(list(p.service_ship_types)).fill_null(False)
               | tb.is_in(list(p.service_ship_types)).fill_null(False))
    return ep.with_columns(
        pair_class=pl.when(service).then(pl.lit("service_involved"))
        .when(focus_a & focus_b).then(pl.lit("cargo_tanker_pair"))
        .when((focus_a & tb.is_null()) | (focus_b & ta.is_null()))
        .then(pl.lit("cargo_tanker_with_unknown"))
        .when((ta == "Fishing") & (tb == "Fishing")).then(pl.lit("fishing_pair"))
        .otherwise(pl.lit("other"))
    )


def flag_dark_context(ep: pl.DataFrame, gaps: pl.DataFrame, window_h: float) -> pl.DataFrame:
    """Mark encounters where either vessel had a 'dark' gap within +-window_h.

    A rendezvous bracketed by AIS silence is the classic sanctions-evasion pattern.
    """
    dark = gaps.filter(pl.col("category") == "dark").select("mmsi", "gap_start", "gap_end")
    w = timedelta(hours=window_h)
    hits = []
    for side in ("a", "b"):
        h = (
            ep.with_row_index("eid").select("eid", pl.col(f"mmsi_{side}").alias("mmsi"),
                                            "start", "end")
            .join(dark, on="mmsi", how="inner")
            .filter((pl.col("gap_end") >= pl.col("start") - w)
                    & (pl.col("gap_start") <= pl.col("end") + w))
            .select("eid")
        )
        hits.append(h)
    hit_ids = pl.concat(hits).unique()["eid"]
    return ep.with_row_index("eid").with_columns(
        dark_nearby=pl.col("eid").is_in(hit_ids.implode())).drop("eid")


def load_candidates(points_paths: list[Path], p: STSParams) -> pl.DataFrame:
    """Slow binned positions from every MMSI bucket (bounded memory)."""
    return pl.concat([
        slow_positions(pl.read_parquet(f, columns=["mmsi", "ts", "lat", "lon", "sog"])
                       .drop_nulls("sog"), p)
        for f in points_paths
    ])
