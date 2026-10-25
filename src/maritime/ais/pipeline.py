"""Part A orchestration, one `run_aX` function per phase.

A1: download -> extract -> stage -> clean -> trajectories. Resumable: a day whose
    cleaned output and stats already exist is skipped unless force=True.
A2: zones -> gaps -> coverage -> classification; manoeuvre events.
A3: slow binned positions -> KD-tree pairs -> encounter episodes.
A4: land test + jumps + circles per cell-hour -> interference flags; A3 coast re-check.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import polars as pl

from maritime.ais.clean import CleaningParams, clean_day, stage_csv
from maritime.ais.dark import (
    DarkParams,
    ManeuverParams,
    add_coverage,
    classify_gaps,
    coverage_table,
    detect_maneuvers,
    find_gaps,
)
from maritime.ais.download import download_day, extract_csv
from maritime.ais.encounters import (
    STSParams,
    build_episodes,
    classify_pairs,
    close_pairs,
    drop_routine,
    flag_dark_context,
    load_candidates,
)
from maritime.ais.gnss_anomaly import (
    CircleParams,
    InterferenceParams,
    convergence,
    distance_to_land_m,
    download_natural_earth,
    flag_cells,
    inland_area,
    load_land,
    process_bucket,
)
from maritime.ais.trajectories import build_trajectories
from maritime.ais.vessels import vessel_registry
from maritime.config import Config
from maritime.plots import (
    dark_gap_map,
    density_map,
    encounter_detail,
    encounter_map,
    gap_category_chart,
    interference_daily_maps,
    interference_timeseries,
)
from maritime.zones import Zones, anchorage_cells, download_wpi, load_wpi


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process_day(day: date, cfg: Config, p: CleaningParams, force: bool = False) -> dict:
    dl, paths = cfg.section("download"), cfg.paths
    clean_dir = paths.interim / "clean"
    pos_out = clean_dir / f"positions_{day}.parquet"
    static_out = clean_dir / f"static_{day}.parquet"
    stats_out = clean_dir / f"stats_{day}.json"

    if not force and pos_out.exists() and stats_out.exists():
        _log(f"{day}: already cleaned, skipping")
        return json.loads(stats_out.read_text())

    stage = paths.interim / "stage" / f"aisdk-{day}.parquet"
    if force or not stage.exists():
        zip_path = download_day(day, paths.raw, dl["base_url"])
        _log(f"{day}: extracting {zip_path.name}")
        csv_path = extract_csv(zip_path, paths.raw / "csv")
        _log(f"{day}: staging CSV -> Parquet")
        stage_csv(csv_path, stage)
        if not dl["keep_csv"]:
            csv_path.unlink()
        if not dl["keep_zip"]:
            zip_path.unlink()

    _log(f"{day}: cleaning")
    t0 = time.perf_counter()
    stats = clean_day(stage, pos_out, static_out, p, cfg.aoi["full"])
    stats = {"day": str(day), **stats, "clean_seconds": round(time.perf_counter() - t0, 1)}
    stats_out.write_text(json.dumps(stats, indent=2))
    _log(f"{day}: {stats['n_raw']:,} raw -> {stats['n_clean']:,} clean fixes, "
         f"{stats['n_vessels']:,} vessels")
    return stats


def run_a1(cfg: Config, days: list[date] | None = None, force: bool = False) -> None:
    cfg.paths.ensure()
    p = CleaningParams.from_dict(cfg.section("cleaning"))
    days = days or cfg.days

    rows = [process_day(d, cfg, p, force) for d in days]
    funnel = pl.DataFrame(rows)
    funnel.write_csv(cfg.paths.tables / "a1_cleaning_funnel.csv")
    _log(f"funnel written:\n{funnel}")

    tp = cfg.section("trajectories")
    _log("building trajectories")
    clean_paths = [cfg.paths.interim / "clean" / f"positions_{d}.parquet" for d in days]
    traj = build_trajectories(clean_paths, cfg.paths.processed / "trajectories", p,
                              tp["max_gap_min"], tp["n_buckets"])
    (cfg.paths.tables / "a1_trajectories.json").write_text(json.dumps(traj, indent=2))
    _log(f"trajectories: {traj}")

    fig = density_map(clean_paths[0], cfg.aoi["full"],
                      cfg.paths.figures / "a1_density.png",
                      title=f"Cleaned AIS fixes, {days[0]} (DMA)")
    _log(f"figure: {fig}")


# ---------------------------------------------------------------- A2

def build_zones(cfg: Config, points_paths: list) -> Zones:
    """Ports (WPI) + data-driven anchorages; cached to data/processed/zones for A3/A4."""
    zc = cfg.section("zones")
    zdir = cfg.paths.processed / "zones"
    zdir.mkdir(parents=True, exist_ok=True)
    wpi = download_wpi(zc["wpi_url"], cfg.paths.reference / "wpi.csv")
    ports = load_wpi(wpi, cfg.aoi["full"], zc["port_radius_km"])
    cells = anchorage_cells(points_paths, zc["anchorage_cell_deg"],
                            zc["anchorage_min_vessels"], zc["anchorage_statuses"])
    ports.write_parquet(zdir / "ports.parquet")
    cells.write_parquet(zdir / "anchorage_cells.parquet")
    _log(f"zones: {ports.height} WPI ports, {cells.height} anchorage cells")
    return Zones(ports, cells, zc["anchorage_cell_deg"])


def run_a2(cfg: Config) -> None:
    traj_dir = cfg.paths.processed / "trajectories"
    points = sorted(traj_dir.glob("points_b*.parquet"))
    segments = pl.read_parquet(traj_dir / "segments.parquet")
    if "end_sog_kn" not in segments.columns:
        raise SystemExit("segments.parquet is from an older A1 run: run `python -m maritime a1` "
                         "first (days are skipped, only trajectories are rebuilt, ~1 min)")

    dp = DarkParams.from_dict(cfg.section("dark"))
    mp = ManeuverParams.from_dict(cfg.section("maneuvers"))
    focus = cfg.section("dark")["focus_ship_types"]
    out_dir = cfg.paths.processed / "dark"
    out_dir.mkdir(parents=True, exist_ok=True)

    zones = build_zones(cfg, points)
    registry = vessel_registry(sorted((cfg.paths.interim / "clean").glob("static_*.parquet")))
    registry.write_parquet(cfg.paths.processed / "vessels.parquet")
    _log(f"vessel registry: {registry.height} MMSIs with static data")

    # ---- gaps
    _log("computing empirical coverage map")
    coverage = coverage_table(points, dp.coverage_cell_deg)
    coverage.write_parquet(cfg.paths.processed / "coverage.parquet")
    gaps = find_gaps(segments, dp.min_gap_h)
    gaps = classify_gaps(add_coverage(gaps, coverage, dp), zones, dp)
    gaps = gaps.join(registry.select("mmsi", "name", "ship_type"), on="mmsi", how="left")
    gaps.write_parquet(out_dir / "gaps.parquet")
    _log(f"gaps >= {dp.min_gap_h} h: {gaps.height}")

    # ---- manoeuvres (per MMSI bucket to bound memory)
    _log("detecting manoeuvres")
    events = []
    for p in points:
        df = pl.read_parquet(p, columns=["mmsi", "seg", "ts", "lat", "lon", "sog", "cog"])
        events.append(detect_maneuvers(df.drop_nulls(["sog", "cog"]), mp))
    events = pl.concat(events)
    events = events.with_columns(in_zone=pl.Series(zones.mask(events)))
    events = events.join(registry.select("mmsi", "name", "ship_type"), on="mmsi", how="left")
    events.write_parquet(out_dir / "maneuvers.parquet")

    summary = summarize_a2(gaps, events, focus)
    (cfg.paths.tables / "a2_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"summary: {json.dumps(summary, indent=2)}")

    dark = gaps.filter(pl.col("category") == "dark")
    top = (dark.filter(pl.col("ship_type").is_in(focus)).sort("gap_h", descending=True)
           .select("mmsi", "name", "ship_type", "moved_while_dark", "gap_start", "gap_end",
                   "gap_h", "start_lat",
                   "start_lon", "end_lat", "end_lon", "gap_dist_nm", "implied_speed_kn",
                   "sog_before_kn", "sog_after_kn", "coverage_start", "coverage_end"))
    top.head(50).write_csv(cfg.paths.tables / "a2_dark_gaps_top.csv")

    figs = cfg.paths.figures
    dark_gap_map(dark, coverage, dp.coverage_cell_deg, cfg.aoi["full"],
                 figs / "a2_dark_gaps_map.png", focus)
    gap_category_chart(gaps, figs / "a2_gap_categories.png")
    _log(f"figures: {figs / 'a2_dark_gaps_map.png'}, {figs / 'a2_gap_categories.png'}")


def summarize_a2(gaps: pl.DataFrame, events: pl.DataFrame, focus: list[str]) -> dict:
    c = pl.col
    dark = gaps.filter(c("category") == "dark")
    dark_focus = dark.filter(c("ship_type").is_in(focus))
    slowed = dark.filter((c("sog_before_kn") > 5)
                         & (c("implied_speed_kn") < 0.5 * c("sog_before_kn")))
    at_sea = events.filter(~c("in_zone"))
    return {
        "gaps_total": gaps.height,
        "gaps_by_category": dict(gaps["category"].value_counts().iter_rows()),
        "dark_gaps": dark.height,
        "dark_vessels": dark["mmsi"].n_unique(),
        "dark_gaps_focus_types": dark_focus.height,
        "dark_vessels_focus_types": dark_focus["mmsi"].n_unique(),
        "dark_gap_median_h": float(dark["gap_h"].median()) if dark.height else None,
        "dark_gaps_moved_while_dark": int(dark["moved_while_dark"].sum()),
        "dark_gaps_moved_focus_types": int(dark_focus["moved_while_dark"].sum()),
        "dark_gaps_slowed_while_silent": slowed.height,
        "dark_by_ship_type": {k or "unknown": v for k, v in
                              dark["ship_type"].value_counts().sort("count", descending=True)
                              .iter_rows()},
        "maneuver_events_total": events.height,
        "maneuver_events_at_sea": dict(at_sea["event"].value_counts().iter_rows()),
        "maneuver_events_at_sea_focus_types": dict(
            at_sea.filter(c("ship_type").is_in(focus))["event"].value_counts().iter_rows()),
    }


# ---------------------------------------------------------------- A3

def run_a3(cfg: Config) -> None:
    proc = cfg.paths.processed
    points = sorted((proc / "trajectories").glob("points_b*.parquet"))
    if not (proc / "zones" / "ports.parquet").exists():
        raise SystemExit("zones not found: run `python -m maritime a2` first")

    p = STSParams.from_dict(cfg.section("sts"))
    focus = cfg.section("dark")["focus_ship_types"]
    zones = Zones.load(proc / "zones", cfg.section("zones")["anchorage_cell_deg"])
    registry = pl.read_parquet(proc / "vessels.parquet")
    gaps = pl.read_parquet(proc / "dark" / "gaps.parquet")
    out_dir = proc / "sts"
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("binning slow positions")
    t0 = time.perf_counter()
    cand = load_candidates(points, p)
    cand, funnel = drop_routine(cand, zones, registry, p)
    _log(f"candidates: {funnel}")

    _log("pairing with KD-tree per time bin")
    t1 = time.perf_counter()
    pairs = close_pairs(cand, p, cfg.crs_metric)
    t_pair = time.perf_counter() - t1
    ep = build_episodes(pairs, p)
    ep = flag_dark_context(classify_pairs(ep, registry, focus, p), gaps, p.dark_window_h)
    ep.write_parquet(out_dir / "encounters.parquet")
    runtime = time.perf_counter() - t0

    summary = {
        **funnel,
        "n_time_bins": cand["bin"].n_unique(),
        "max_candidates_in_one_bin": int(cand.group_by("bin").len()["len"].max()),
        "close_pair_bins": pairs.height,
        "encounters": ep.height,
        "encounters_by_class": dict(ep["pair_class"].value_counts().iter_rows()),
        "encounters_with_dark_nearby": int(ep["dark_nearby"].sum()),
        "focus_encounters_with_dark_nearby": int(
            ep.filter(pl.col("pair_class").str.starts_with("cargo_tanker"))["dark_nearby"].sum()),
        "median_duration_h": float(ep["duration_h"].median()) if ep.height else None,
        "pairing_seconds": round(t_pair, 1),
        "total_seconds": round(runtime, 1),
    }
    (cfg.paths.tables / "a3_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"summary: {json.dumps(summary, indent=2)}")

    cols = ["pair_class", "dark_nearby", "mmsi_a", "name_a", "ship_type_a", "mmsi_b", "name_b",
            "ship_type_b", "start", "end", "duration_h", "mean_dist_m", "min_dist_m",
            "lat", "lon"]
    focus_ep = ep.filter(pl.col("pair_class").str.starts_with("cargo_tanker"))
    (focus_ep.sort(["dark_nearby", "duration_h"], descending=True).select(cols)
     .write_csv(cfg.paths.tables / "a3_encounters_cargo_tanker.csv"))

    figs = cfg.paths.figures
    coverage = pl.read_parquet(proc / "coverage.parquet")
    encounter_map(ep, coverage, cfg.section("dark")["coverage_cell_deg"], cfg.aoi["full"],
                  figs / "a3_encounters_map.png")
    # Illustrative example: an operation-length encounter. Multi-day "encounters" are almost
    # always two ships moored together in a small harbour missing from WPI.
    example = focus_ep.filter(pl.col("duration_h") <= cfg.section("sts")["example_max_h"])
    if example.height:
        top = example.sort(["dark_nearby", "duration_h"], descending=True).row(0, named=True)
        w = timedelta(hours=3)
        tracks = pl.concat([
            pl.scan_parquet(f).filter(pl.col("mmsi").is_in([top["mmsi_a"], top["mmsi_b"]])
                                      & pl.col("ts").is_between(top["start"] - w, top["end"] + w))
            .select("mmsi", "ts", "lat", "lon").collect()
            for f in points])
        encounter_detail(tracks, top, figs / "a3_encounter_example.png")
    _log(f"figures in {figs}")


# ---------------------------------------------------------------- A4

def run_a4(cfg: Config) -> None:
    proc, tables, figs = cfg.paths.processed, cfg.paths.tables, cfg.paths.figures
    points = sorted((proc / "trajectories").glob("points_b*.parquet"))
    if not (proc / "sts" / "encounters.parquet").exists():
        raise SystemExit("A3 output not found: run `python -m maritime a3` first")

    ic = cfg.section("interference")
    p = InterferenceParams.from_dict(ic)
    cp = CircleParams.from_dict(cfg.section("circles"))
    zones = Zones.load(proc / "zones", cfg.section("zones")["anchorage_cell_deg"])
    registry = pl.read_parquet(proc / "vessels.parquet")
    out_dir = proc / "gnss"
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("loading Natural Earth land polygons")
    ne = download_natural_earth(ic["natural_earth_url"], ic["natural_earth_layers"],
                                cfg.paths.reference / "ne")
    land = load_land(ne, cfg.aoi["full"], cfg.crs_metric)
    inland = inland_area(land, p.inland_buffer_m)

    _log("scanning MMSI buckets (land test, jumps, circles)")
    t0 = time.perf_counter()
    stats, dests, circles, n_land, n_land_jump = [], [], [], 0, 0
    for f in points:
        s, d, c, nl, nlj = process_bucket(f, inland, cfg.crs_metric, zones,
                                          cfg.section("sts")["zone_cell_deg"], p, cp)
        stats.append(s)
        dests.append(d)
        circles.append(c)
        n_land += nl
        n_land_jump += nlj
    cells = flag_cells(pl.concat(stats), p)
    conv = convergence(pl.concat(dests), p)
    circles = pl.concat(circles).join(registry.select("mmsi", "name", "ship_type"),
                                      on="mmsi", how="left")
    circles = circles.with_columns(
        unexpected=~pl.col("ship_type").is_in(list(cp.expected_types)).fill_null(False))
    cells.write_parquet(out_dir / "interference_cells.parquet")
    conv.write_parquet(out_dir / "convergence.parquet")
    circles.write_parquet(out_dir / "circles.parquet")
    scan_s = time.perf_counter() - t0

    # ---- regional time series
    lon_min, lat_min, lon_max, lat_max = cfg.aoi["bornholm"]
    c = pl.col
    in_region = (((c("cx") + 0.5) * p.cell_deg).is_between(lon_min, lon_max)
                 & ((c("cy") + 0.5) * p.cell_deg).is_between(lat_min, lat_max))
    hours = pl.DataFrame({"hour": pl.datetime_range(cells["hour"].min(), cells["hour"].max(),
                                                    "1h", eager=True)})
    agg = cells.with_columns(region=in_region).group_by("hour").agg(
        affected_region=c("affected").filter(c("flagged") & c("region")).sum(),
        affected_rest=c("affected").filter(c("flagged") & ~c("region")).sum(),
        affected_region_all=c("affected").filter(c("region")).sum(),
        n_region=c("n").filter(c("region")).sum(),
    )
    hourly = hours.join(agg, on="hour", how="left").fill_null(0).with_columns(
        share_region=pl.when(c("n_region") > 0)
        .then(c("affected_region_all") / c("n_region")).otherwise(0.0)).sort("hour")
    hourly.write_csv(tables / "a4_hourly_bornholm_vs_rest.csv")

    # ---- A3 re-check: encounters right next to the coastline are berths, not STS
    enc = pl.read_parquet(proc / "sts" / "encounters.parquet")
    enc = enc.with_columns(dist_to_land_m=pl.Series(
        distance_to_land_m(enc["lon"].to_numpy(), enc["lat"].to_numpy(), land, cfg.crs_metric)))
    enc = enc.with_columns(near_coast=c("dist_to_land_m") < ic["near_coast_m"])
    enc.write_parquet(proc / "sts" / "encounters_coast.parquet")
    focus_enc = enc.filter(c("pair_class").str.starts_with("cargo_tanker"))

    flagged = cells.filter(c("flagged"))
    by_day = (flagged.group_by(c("hour").dt.date().alias("day"))
              .agg(cell_hours=pl.len(), vessel_hours=c("affected").sum()).sort("day"))
    peak = flagged.sort("affected", descending=True).head(1)
    summary = {
        "scan_seconds": round(scan_s, 1),
        "fixes_deep_inland_outside_ports": n_land,
        "fixes_on_land_after_jump": n_land_jump,
        "cell_hours_total": cells.height,
        "cell_hours_flagged": flagged.height,
        "flagged_by_day": {str(r["day"]): {"cell_hours": r["cell_hours"],
                                           "affected_vessel_hours": r["vessel_hours"]}
                           for r in by_day.iter_rows(named=True)},
        "flagged_cell_hours_in_bornholm_box": int(flagged.filter(in_region).height),
        "peak_cell_hour": {k: str(v) for k, v in peak.row(0, named=True).items()}
        if peak.height else None,
        "peak_hour_bornholm_affected": int(hourly["affected_region"].max()),
        "peak_hour_bornholm_share_pct": round(100 * float(hourly["share_region"].max()), 1),
        "convergent_jump_cells": conv.height,
        "circle_events": circles.height,
        "circle_events_unexpected_types": int(circles["unexpected"].sum()),
        "a3_encounters_near_coast": int(enc["near_coast"].sum()),
        "a3_cargo_tanker_encounters": focus_enc.height,
        "a3_cargo_tanker_near_coast": int(focus_enc["near_coast"].sum()),
    }
    (tables / "a4_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"summary: {json.dumps(summary, indent=2)}")
    (focus_enc.select("pair_class", "name_a", "name_b", "start", "duration_h", "lat", "lon",
                      "dist_to_land_m", "near_coast").sort("dist_to_land_m")
     .write_csv(tables / "a4_a3_cargo_tanker_coast_check.csv"))
    (circles.filter(c("unexpected")).sort("ts")
     .write_csv(tables / "a4_circles_unexpected_types.csv"))

    coverage = pl.read_parquet(proc / "coverage.parquet")
    interference_daily_maps(cells, coverage, p.cell_deg, cfg.section("dark")["coverage_cell_deg"],
                            cfg.aoi["full"],
                            figs / "a4_interference_daily.png")
    interference_timeseries(hourly, "Bornholm box", figs / "a4_interference_timeseries.png")
    _log(f"figures in {figs}")
