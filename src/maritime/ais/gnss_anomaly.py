"""A4 - GNSS interference indicators in AIS.

AIS positions come from each ship's own GNSS receiver, so jamming or spoofing shows up in
the AIS stream. One bad receiver is a fault; many independent receivers going wrong in the
same place and hour is interference. Indicators, all counted as DISTINCT vessels per
(grid cell, hour). Cells are deliberately coarse (0.5 deg ~ 55 x 31 km): a jammer
affects tens to hundreds of km, so finer cells split one event into many sub-threshold
pieces.

  jump        physically impossible move between consecutive fixes (from A1)
  land        fix more than `inland_buffer_m` inside Natural Earth land, outside port zones,
              in a track segment that BEGAN WITH a jump. Narrow fjords and sounds that a
              1:10M map closes are reached smoothly by real traffic; a receiver displaced
              onto land arrives there by an impossible jump.
  convergence several vessels jumping INTO the same small cell in the same hour: the
              signature of a spoofer broadcasting one fake position. Scattered jump
              destinations instead point to jamming / degraded receivers.
  circle      >= 360 deg of turning within a short window by a vessel type that does not
              normally circle (cargo, tanker...): a known spoofing artefact.

A cell-hour is flagged when >= `min_affected_vessels` vessels AND >= `min_affected_share`
of the vessels present there were affected (jump or land).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import requests
import shapely

from maritime.geo import angle_diff_expr, to_metric
from maritime.zones import Zones, cell_exprs


@dataclass(frozen=True)
class InterferenceParams:
    inland_buffer_m: float
    cell_deg: float
    min_affected_vessels: int
    min_affected_share: float
    convergence_cell_deg: float
    convergence_min_vessels: int

    @classmethod
    def from_dict(cls, d: dict) -> InterferenceParams:
        return cls(float(d["inland_buffer_m"]), float(d["cell_deg"]),
                   int(d["min_affected_vessels"]),
                   float(d["min_affected_share"]), float(d["convergence_cell_deg"]),
                   int(d["convergence_min_vessels"]))


@dataclass(frozen=True)
class CircleParams:
    window_min: float
    min_turn_deg: float
    min_sog_kn: float
    max_dt_s: float
    merge_min: float
    expected_types: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: dict) -> CircleParams:
        return cls(float(d["window_min"]), float(d["min_turn_deg"]), float(d["min_sog_kn"]),
                   float(d["max_dt_s"]), float(d["merge_min"]), tuple(d["expected_types"]))


# ---------------------------------------------------------------- land polygons

def download_natural_earth(base_url: str, layers: list[str], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for layer in layers:
        p = dest_dir / f"{layer}.zip"
        if not p.exists():
            r = requests.get(f"{base_url.rstrip('/')}/{layer}.zip", timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
        out.append(p)
    return out


def load_land(paths: list[Path], bbox, crs: str):
    """Union of land polygons around bbox, projected to the metric CRS."""
    import geopandas as gpd

    lon_min, lat_min, lon_max, lat_max = bbox
    read_box = (lon_min - 1, lat_min - 1, lon_max + 1, lat_max + 1)
    parts = [gpd.read_file(p, bbox=read_box) for p in paths]
    # The bbox filter returns whole intersecting polygons (all of Eurasia). Clip in degrees
    # first: projecting Siberia into UTM 32N distorts it into invalid geometry.
    geoms = shapely.make_valid(shapely.clip_by_rect(
        np.array([g for df in parts for g in df.geometry]), *read_box))
    land = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326").to_crs(crs)
    return shapely.make_valid(shapely.union_all(land.geometry.values))


def inland_area(land, buffer_m: float):
    """Land shrunk by buffer_m: a fix inside this is unambiguously on land, whatever the
    ~km-level coastline error of a 1:10M map. Prepared for fast point tests."""
    inner = land.buffer(-buffer_m)
    shapely.prepare(inner)
    return inner


def points_on_land(lon: np.ndarray, lat: np.ndarray, inland, crs: str) -> np.ndarray:
    x, y = to_metric(np.asarray(lon), np.asarray(lat), crs)
    return shapely.contains_xy(inland, x, y)


def distance_to_land_m(lon: np.ndarray, lat: np.ndarray, land, crs: str) -> np.ndarray:
    x, y = to_metric(np.asarray(lon), np.asarray(lat), crs)
    return shapely.distance(land, shapely.points(x, y))


# ---------------------------------------------------------------- per-bucket indicators

def landed_after_jump(df: pl.DataFrame, on_land: np.ndarray) -> np.ndarray:
    """On-land fixes whose track segment started with an impossible jump.

    Requires the A1 trajectory columns mmsi, seg, jump (a jump always opens a new segment,
    so the first fix of a jump-born segment carries jump=True).
    """
    seg_jumped = df.select(pl.col("jump").first().over(["mmsi", "seg"])).to_series()
    return on_land & seg_jumped.to_numpy()


def cell_hour_stats(df: pl.DataFrame, land: np.ndarray, p: InterferenceParams
                    ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Distinct-vessel counts per (cell, hour) and jump-destination counts per small cell.

    `df` holds one MMSI bucket (columns mmsi, ts, lat, lon, jump); because buckets partition
    MMSIs, distinct counts from different buckets can simply be summed.
    """
    c = pl.col
    d = df.with_columns(*cell_exprs(p.cell_deg), hour=c("ts").dt.truncate("1h"),
                        land=pl.Series(land))
    stats = d.group_by(["cx", "cy", "hour"]).agg(
        n=c("mmsi").n_unique(),
        jump_vessels=c("mmsi").filter(c("jump")).n_unique(),
        land_vessels=c("mmsi").filter(c("land")).n_unique(),
        affected=c("mmsi").filter(c("jump") | c("land")).n_unique(),
    )
    dest = (
        d.filter(c("jump"))
        .with_columns(cell_exprs(p.convergence_cell_deg, prefix="d"))
        .group_by(["dcx", "dcy", "hour"]).agg(nv=c("mmsi").n_unique())
    )
    return stats, dest


def flag_cells(stats: pl.DataFrame, p: InterferenceParams) -> pl.DataFrame:
    """Sum bucket counts per cell-hour and flag where enough vessels are affected."""
    c = pl.col
    return (
        stats.group_by(["cx", "cy", "hour"])
        .agg(c("n", "jump_vessels", "land_vessels", "affected").sum())
        .with_columns(share=c("affected") / c("n"))
        .with_columns(flagged=(c("affected") >= p.min_affected_vessels)
                      & (c("share") >= p.min_affected_share))
        .sort(["hour", "cx", "cy"])
    )


def convergence(dest: pl.DataFrame, p: InterferenceParams) -> pl.DataFrame:
    """Small cells receiving jumps from >= convergence_min_vessels distinct vessels in an hour."""
    return (dest.group_by(["dcx", "dcy", "hour"]).agg(pl.col("nv").sum())
                .filter(pl.col("nv") >= p.convergence_min_vessels)
                .with_columns(lat=(pl.col("dcy") + 0.5) * p.convergence_cell_deg,
                              lon=(pl.col("dcx") + 0.5) * p.convergence_cell_deg)
                .sort("nv", descending=True))


# ---------------------------------------------------------------- circles

def detect_circles(df: pl.DataFrame, p: CircleParams) -> pl.DataFrame:
    """Windows where a vessel's course turned >= min_turn_deg in total within window_min.

    Signed course changes are summed over a trailing time window per (mmsi, seg); steps
    with a long time gap or low speed are zeroed (COG is noise near 0 kn).
    """
    c = pl.col
    same = (c("mmsi") == c("mmsi").shift(1)) & (c("seg") == c("seg").shift(1))
    dt = (c("ts") - c("ts").shift(1)).dt.total_seconds()
    d = df.sort(["mmsi", "seg", "ts"]).with_columns(
        dcog=pl.when(same & (dt <= p.max_dt_s) & (c("sog") >= p.min_sog_kn)
                     & (c("sog").shift(1) >= p.min_sog_kn))
        .then(angle_diff_expr(c("cog"), c("cog").shift(1))).otherwise(0.0)
    ).with_columns(
        turn=c("dcog").rolling_sum_by("ts", window_size=f"{int(p.window_min)}m")
        .over(["mmsi", "seg"])
    )
    hits = d.filter(c("turn").abs() >= p.min_turn_deg).sort(["mmsi", "ts"])
    if hits.is_empty():
        return pl.DataFrame(schema={"mmsi": pl.Int64, "ts": pl.Datetime("us", "UTC"),
                                    "lat": pl.Float64, "lon": pl.Float64,
                                    "turn": pl.Float64, "n_fixes": pl.UInt32})
    merge = timedelta(minutes=p.merge_min)
    new = ((c("mmsi") != c("mmsi").shift(1))
           | ((c("ts") - c("ts").shift(1)) > merge)).fill_null(True)
    return (hits.with_columns(eid=new.cum_sum())
                .group_by("eid", maintain_order=True)
                .agg(c("mmsi", "ts", "lat", "lon").first(), c("turn").abs().max(),
                     n_fixes=pl.len())
                .drop("eid"))


# ---------------------------------------------------------------- orchestration helper

def process_bucket(path: Path, inland, crs: str, zones: Zones, zone_cell_deg: float,
                   p: InterferenceParams, cp: CircleParams):
    df = pl.read_parquet(path, columns=["mmsi", "seg", "ts", "lat", "lon", "sog", "cog", "jump"])
    on_land = points_on_land(df["lon"].to_numpy(), df["lat"].to_numpy(), inland, crs)
    if on_land.any():  # ports sit on the coastline: never count a fix inside a port zone
        idx = np.flatnonzero(on_land)
        on_land[idx[zones.mask_gridded(df[idx], zone_cell_deg)]] = False
    land = landed_after_jump(df, on_land)
    stats, dest = cell_hour_stats(df, land, p)
    circles = detect_circles(df.drop_nulls(["sog", "cog"]), cp)
    if circles.height:  # ferries turning in harbour basins are routine
        circles = circles.filter(~pl.Series(zones.mask_gridded(circles, zone_cell_deg)))
    return stats, dest, circles, int(on_land.sum()), int(land.sum())
