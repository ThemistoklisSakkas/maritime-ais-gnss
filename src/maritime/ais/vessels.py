"""One row per MMSI with the best available static identity (name, type, size).

AIS static data is typed by the crew and repeats every few minutes, so a vessel can
appear with several spellings or an empty type. We take the most frequently reported
non-null value of each field over the study period.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _most_common(col: str) -> pl.Expr:
    # mode() can return several tied values in arbitrary order: sort for reproducibility
    return pl.col(col).drop_nulls().mode().sort().first().alias(col)


def vessel_registry(static_paths: list[Path]) -> pl.DataFrame:
    s = pl.concat([pl.read_parquet(p) for p in static_paths], how="diagonal_relaxed")
    return (
        s.with_columns(pl.col("ship_type").cast(pl.String))
        .group_by("mmsi")
        .agg(_most_common("name"), _most_common("ship_type"), _most_common("imo"),
             _most_common("callsign"), _most_common("length"))
        .sort("mmsi")
    )
