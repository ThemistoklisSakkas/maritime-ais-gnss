"""Phase A1 orchestration: download -> extract -> stage -> clean -> trajectories.

Every step is resumable: a day whose cleaned output and stats already exist is
skipped unless force=True, so a crash on day 4 does not redo days 1-3.
"""

from __future__ import annotations

import json
import time
from datetime import date

import polars as pl

from maritime.ais.clean import CleaningParams, clean_day, stage_csv
from maritime.ais.download import download_day, extract_csv
from maritime.ais.trajectories import build_trajectories
from maritime.config import Config
from maritime.plots import density_map


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
