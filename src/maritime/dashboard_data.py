"""Part C - Export a small, committed data bundle for the dashboard.

The raw and processed AIS data (several GB) stay out of git. The dashboard only needs a
few thousand rows per view, so they are copied to `outputs/dashboard/` (a few MB, committed),
and anyone who clones the repository can open the dashboard without re-running Part A.
"""

from __future__ import annotations

import json

import polars as pl

from maritime.config import Config


def export_dashboard_data(cfg: Config) -> dict:
    proc, tables = cfg.paths.processed, cfg.paths.tables
    out = cfg.paths.tables.parent / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    c = pl.col
    sizes = {}

    gaps = pl.read_parquet(proc / "dark" / "gaps.parquet").filter(c("category") == "dark")
    gaps.select("mmsi", "name", "ship_type", "gap_start", "gap_end", "gap_h", "start_lat",
                "start_lon", "end_lat", "end_lon", "gap_dist_nm", "sog_before_kn",
                "sog_after_kn", "moved_while_dark").write_parquet(out / "dark_gaps.parquet")
    sizes["dark_gaps"] = gaps.height

    enc = pl.read_parquet(proc / "sts" / "encounters_coast.parquet")
    enc.select("mmsi_a", "name_a", "ship_type_a", "mmsi_b", "name_b", "ship_type_b", "start",
               "end", "duration_h", "mean_dist_m", "min_dist_m", "lat", "lon", "pair_class",
               "dark_nearby", "dist_to_land_m", "near_coast") \
       .write_parquet(out / "encounters.parquet")
    sizes["encounters"] = enc.height

    cell = float(cfg.section("interference")["cell_deg"])
    cells = pl.read_parquet(proc / "gnss" / "interference_cells.parquet").with_columns(
        lat=(c("cy") + 0.5) * cell, lon=(c("cx") + 0.5) * cell)
    cells.write_parquet(out / "interference_cells.parquet")
    sizes["interference_cells"] = cells.height

    ports = pl.read_parquet(proc / "zones" / "ports.parquet").with_row_index("port_idx") \
              .select(c("port_idx").cast(pl.Int64), c("port").alias("port_name"))
    eta = (pl.read_parquet(proc / "eta" / "test_predictions.parquet")
             .join(ports, left_on="port", right_on="port_idx", how="left")
             .select("arrival_id", "mmsi", "port_name", "arrival", "t", "target_h",
                     "baseline_h", "model_h", "crew_h", "dist_nm", "sog_kn", "lat", "lon",
                     "port_lat", "port_lon", "destination"))
    eta.write_parquet(out / "eta_test.parquet")
    sizes["eta_samples"] = eta.height

    (out / "manifest.json").write_text(json.dumps(
        {"rows": sizes, "source_tables": sorted(p.name for p in tables.glob("*.json"))},
        indent=2))
    return sizes
