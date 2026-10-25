"""Port and anchorage zones: where stopping, loitering or switching AIS off is routine.

Two complementary sources:
  1. World Port Index (NGA Pub. 150): official port positions, buffered by a radius that
     grows with harbour size. Positions are only given to ~1 arc-minute, hence the buffer.
  2. Data-driven anchorages: grid cells where several *different* vessels reported
     "At anchor" or "Moored" during the study period. WPI has no anchorage polygons, and
     many Danish anchorages lie well outside any port radius.

Used by A2 (dark gaps), A3 (STS encounters) and A4 (interference) to exclude areas where
the behaviour we look for is normal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import requests

from maritime.config import BBox
from maritime.geo import EARTH_RADIUS_M

# ---------------------------------------------------------------- grid cells

def cell_exprs(cell_deg: float, lat: str = "lat", lon: str = "lon",
               prefix: str = "") -> list[pl.Expr]:
    """Integer grid-cell indices (cx, cy) for a regular lat/lon grid of size cell_deg."""
    return [
        (pl.col(lon) / cell_deg).floor().cast(pl.Int32).alias(f"{prefix}cx"),
        (pl.col(lat) / cell_deg).floor().cast(pl.Int32).alias(f"{prefix}cy"),
    ]


# ---------------------------------------------------------------- World Port Index

def download_wpi(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_wpi(path: Path, bbox: BBox, radius_km: dict[str, float]) -> pl.DataFrame:
    """Ports inside bbox with an exclusion radius chosen from WPI 'Harbor Size'."""
    lon_min, lat_min, lon_max, lat_max = bbox
    default = radius_km["default"]
    df = pl.read_csv(path, infer_schema=False, encoding="utf8-lossy")
    return (
        df.select(
            pl.col("Main Port Name").alias("port"),
            pl.col("UN/LOCODE").alias("locode"),
            pl.col("Latitude").cast(pl.Float64).alias("lat"),
            pl.col("Longitude").cast(pl.Float64).alias("lon"),
            pl.col("Harbor Size").fill_null("").alias("size"),
        )
        .filter(pl.col("lat").is_between(lat_min, lat_max)
                & pl.col("lon").is_between(lon_min, lon_max))
        .with_columns(
            pl.col("size").replace_strict(radius_km, default=default, return_dtype=pl.Float64)
            .alias("radius_km")
        )
    )


def near_port(lat: np.ndarray, lon: np.ndarray, ports: pl.DataFrame,
              chunk: int = 20_000) -> np.ndarray:
    """True where a point lies within its nearest-qualifying port radius.

    Brute force (points x ports) in chunks: with ~150 ports this is cheaper and simpler
    than a spatial index, and exact on the sphere.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    out = np.zeros(lat.shape, dtype=bool)
    if ports.is_empty() or lat.size == 0:
        return out
    plat = np.radians(ports["lat"].to_numpy())[None, :]
    plon = np.radians(ports["lon"].to_numpy())[None, :]
    prad = ports["radius_km"].to_numpy()[None, :] * 1000.0
    for i in range(0, lat.size, chunk):
        la = np.radians(lat[i:i + chunk])[:, None]
        lo = np.radians(lon[i:i + chunk])[:, None]
        a = (np.sin((plat - la) / 2) ** 2
             + np.cos(la) * np.cos(plat) * np.sin((plon - lo) / 2) ** 2)
        d = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        out[i:i + chunk] = (d <= prad).any(axis=1)
    return out


# ---------------------------------------------------------------- data-driven anchorages

def anchorage_cells(points_paths: list[Path], cell_deg: float, min_vessels: int,
                    statuses: list[str]) -> pl.DataFrame:
    """Cells (cx, cy) where >= min_vessels distinct MMSIs reported an anchored/moored status.

    Requiring several different vessels keeps a single pair of ships stopped at sea (an
    STS candidate) from turning its own location into an "anchorage".
    """
    per_file = []
    for p in points_paths:  # files are MMSI buckets, so distinct counts add up across them
        per_file.append(
            pl.scan_parquet(p)
            .filter(pl.col("nav_status").cast(pl.String).is_in(statuses))
            .with_columns(cell_exprs(cell_deg))
            .group_by(["cx", "cy"])
            .agg(n=pl.col("mmsi").n_unique())
            .collect()
        )
    if not per_file:
        return pl.DataFrame(schema={"cx": pl.Int32, "cy": pl.Int32})
    cells = pl.concat(per_file).group_by(["cx", "cy"]).agg(pl.col("n").sum())
    return cells.filter(pl.col("n") >= min_vessels).select("cx", "cy")


def in_anchorage(df: pl.DataFrame, cells: pl.DataFrame, cell_deg: float,
                 lat: str = "lat", lon: str = "lon") -> np.ndarray:
    """True where (lat, lon) falls in one of the anchorage cells."""
    keyed = df.select(cell_exprs(cell_deg, lat, lon)).with_row_index("_i")
    hit = keyed.join(cells.with_columns(_hit=pl.lit(True)), on=["cx", "cy"], how="left")
    return hit.sort("_i")["_hit"].fill_null(False).to_numpy()


class Zones:
    """Ports + anchorages bundled, so callers ask a single question: routine area or not."""

    def __init__(self, ports: pl.DataFrame, anchor_cells: pl.DataFrame, anchor_cell_deg: float):
        self.ports = ports
        self.anchor_cells = anchor_cells
        self.anchor_cell_deg = anchor_cell_deg

    def mask(self, df: pl.DataFrame, lat: str = "lat", lon: str = "lon") -> np.ndarray:
        if df.is_empty():
            return np.zeros(0, dtype=bool)
        port = near_port(df[lat].to_numpy(), df[lon].to_numpy(), self.ports)
        anch = in_anchorage(df, self.anchor_cells, self.anchor_cell_deg, lat, lon)
        return port | anch
