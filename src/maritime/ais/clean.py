"""Parse, validate and clean one day of DMA AIS data.

Pipeline per day (memory-bounded, one day at a time):

    raw CSV  --stage_csv-->  stage Parquet (all rows, typed, renamed)
    stage    --clean_day-->  positions Parquet (vessel position reports, cleaned)
                             static Parquet    (distinct voyage/static records)

Design rule: cleaning removes *noise* (invalid fields, duplicates, isolated
position spikes) but keeps *signals*. A track that jumps and stays at the new
location is flagged (`jump=True`), not deleted, because that is exactly what
GNSS spoofing looks like and phase A4 needs to see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from maritime.config import BBox
from maritime.geo import KNOT_MS, haversine_expr

RAW_COLUMNS = {
    "# Timestamp": "ts",
    "Type of mobile": "mobile_type",
    "MMSI": "mmsi",
    "Latitude": "lat",
    "Longitude": "lon",
    "Navigational status": "nav_status",
    "ROT": "rot",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "IMO": "imo",
    "Callsign": "callsign",
    "Name": "name",
    "Ship type": "ship_type",
    "Cargo type": "cargo_type",
    "Width": "width",
    "Length": "length",
    "Draught": "draught",
    "Destination": "destination",
    "ETA": "eta",
}

# DMA writes these placeholders instead of leaving a field empty
NULL_TOKENS = ["", "Unknown", "Undefined", "Unknown value"]

POSITION_COLS = ["ts", "mmsi", "mobile_type", "lat", "lon", "sog", "cog", "heading", "rot",
                 "nav_status"]
STATIC_COLS = ["imo", "callsign", "name", "ship_type", "cargo_type", "width", "length",
               "draught", "destination", "eta"]


@dataclass(frozen=True)
class CleaningParams:
    mobile_types: tuple[str, ...]
    mmsi_min: int
    mmsi_max: int
    max_sog_kn: float
    max_speed_kn: float
    min_jump_m: float
    spike_passes: int
    stale_min_sog_kn: float
    low_res_tol: float

    @classmethod
    def from_dict(cls, d: dict) -> CleaningParams:
        return cls(
            mobile_types=tuple(d["mobile_types"]),
            mmsi_min=int(d["mmsi_min"]),
            mmsi_max=int(d["mmsi_max"]),
            max_sog_kn=float(d["max_sog_kn"]),
            max_speed_kn=float(d["max_speed_kn"]),
            min_jump_m=float(d["min_jump_m"]),
            spike_passes=int(d["spike_passes"]),
            stale_min_sog_kn=float(d["stale_min_sog_kn"]),
            low_res_tol=float(d["low_res_tol"]),
        )


# ---------------------------------------------------------------- parsing

def _str(col: str) -> pl.Expr:
    s = pl.col(col).str.strip_chars()
    return pl.when(s.is_in(NULL_TOKENS)).then(None).otherwise(s).alias(col)


def _num(col: str, dtype: pl.DataType) -> pl.Expr:
    return _str(col).cast(dtype, strict=False)


def scan_raw_csv(path: Path) -> pl.LazyFrame:
    """Lazily read a DMA CSV with every column as text, then rename and type-cast.

    Reading as text first means one malformed value becomes a null instead of
    aborting a 26-million-row parse.
    """
    lf = pl.scan_csv(path, infer_schema=False, encoding="utf8-lossy")
    lf = lf.rename(RAW_COLUMNS, strict=False)
    return lf.select(
        pl.col("ts").str.strptime(pl.Datetime("us"), "%d/%m/%Y %H:%M:%S", strict=False)
        .dt.replace_time_zone("UTC"),
        _str("mobile_type").cast(pl.Categorical),
        _num("mmsi", pl.Int64),
        _num("lat", pl.Float64),
        _num("lon", pl.Float64),
        _str("nav_status").cast(pl.Categorical),
        _num("rot", pl.Float32),
        _num("sog", pl.Float32),
        _num("cog", pl.Float32),
        _num("heading", pl.Float32),
        _num("imo", pl.Int64),
        _str("callsign"),
        _str("name"),
        _str("ship_type").cast(pl.Categorical),
        _str("cargo_type"),
        _num("width", pl.Float32),
        _num("length", pl.Float32),
        _num("draught", pl.Float32),
        _str("destination"),
        _str("eta"),
    )


def stage_csv(csv_path: Path, stage_path: Path) -> Path:
    """CSV -> typed Parquet, streamed (never holds the whole day in RAM)."""
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    scan_raw_csv(csv_path).sink_parquet(stage_path)
    return stage_path


# ---------------------------------------------------------------- validity rules

def is_vessel(p: CleaningParams) -> pl.Expr:
    """Class A / Class B transponders only (drops base stations, AtoN, SAR aircraft)."""
    return pl.col("mobile_type").cast(pl.String).is_in(list(p.mobile_types))


def valid_mmsi(p: CleaningParams) -> pl.Expr:
    """Ship-station MMSI: 9 digits with a Maritime Identification Digit 201..775.

    Excludes coast stations (00x), SAR aircraft (111), AtoN (99x), craft associated
    with a parent ship (98x), SART/MOB/EPIRB (97x) and common default/garbage values
    such as 123456789 (MID 123 < 201). 999999999 is out of range already.
    """
    m = pl.col("mmsi")
    return m.is_between(p.mmsi_min, p.mmsi_max)


def valid_position() -> pl.Expr:
    """Inside WGS84 range, not the AIS 'not available' sentinels (91, 181), not (0, 0)."""
    lat, lon = pl.col("lat"), pl.col("lon")
    return (
        pl.col("ts").is_not_null()
        & lat.is_between(-90, 90) & lon.is_between(-180, 180)
        & ~((lat == 0) & (lon == 0))
    )


def in_bbox(bbox: BBox) -> pl.Expr:
    lon_min, lat_min, lon_max, lat_max = bbox
    return pl.col("lon").is_between(lon_min, lon_max) & pl.col("lat").is_between(lat_min, lat_max)


def normalize_sentinels(p: CleaningParams) -> list[pl.Expr]:
    """Turn AIS 'not available' codes into nulls (102.3 kn SOG, 360 COG, 511 heading, ROT +-720)."""
    sog, cog, hdg, rot = (pl.col(c) for c in ("sog", "cog", "heading", "rot"))
    return [
        pl.when(sog.is_between(0, p.max_sog_kn)).then(sog).alias("sog"),
        pl.when((cog >= 0) & (cog < 360)).then(cog).alias("cog"),
        pl.when((hdg >= 0) & (hdg < 360)).then(hdg).alias("heading"),
        pl.when(rot.abs() < 720).then(rot).alias("rot"),
    ]


# ---------------------------------------------------------------- report-level artefacts

def is_low_resolution(p: CleaningParams) -> pl.Expr:
    """Position lies on the 1/10 arc-minute grid in BOTH lat and lon.

    Standard position reports carry 1/10000 arc-minute; long-range broadcasts
    (AIS message 27) only 1/10 arc-minute (~185 m). Mixed into a dense track they
    create fake 100-200 m zig-zags. A regular fix lands on that grid in both
    axes by chance with probability ~(2*tol)^2, i.e. ~4e-6 for tol=1e-3.
    """
    def on_grid(col: str) -> pl.Expr:
        x = pl.col(col) * 600.0  # degrees -> tenths of arc-minute
        return (x - x.round()).abs() < p.low_res_tol

    return on_grid("lat") & on_grid("lon")


def drop_low_resolution(df: pl.DataFrame, p: CleaningParams) -> tuple[pl.DataFrame, int]:
    mask = df.select(is_low_resolution(p)).to_series()
    return df.filter(~mask), int(mask.sum())


def drop_stale_repeats(df: pl.DataFrame, p: CleaningParams) -> tuple[pl.DataFrame, int]:
    """Drop a fix that repeats the previous position exactly while the ship reports way on.

    A vessel doing 10 kn cannot be at the identical 1e-6 deg position seconds
    later: the repeat is an old fix re-stamped with a newer time. Kept as-is it
    makes the *next* real fix look like a jump. Stationary vessels (SOG below
    `stale_min_sog_kn` or unknown) keep their repeats so they do not develop
    artificial gaps. Requires df sorted by (mmsi, ts).
    """
    c = pl.col
    stale = (
        (c("mmsi") == c("mmsi").shift(1))
        & (c("lat") == c("lat").shift(1))
        & (c("lon") == c("lon").shift(1))
        & (c("sog") > p.stale_min_sog_kn)
    ).fill_null(False)
    mask = df.select(stale).to_series()
    return df.filter(~mask), int(mask.sum())


# ---------------------------------------------------------------- kinematic screening

def _jump_between(lat1, lon1, ts1, lat2, lon2, ts2, p: CleaningParams) -> pl.Expr:
    """True if moving between two fixes needs > max_speed_kn over > min_jump_m.

    The distance floor stops GNSS noise over 1-2 s intervals (10-30 m) from
    looking like a 40 kn sprint. dt is floored at 1 s (timestamp resolution).
    """
    d = haversine_expr(lat1, lon1, lat2, lon2)
    dt = (ts2 - ts1).dt.total_seconds().clip(lower_bound=1)
    return (d > p.min_jump_m) & (d / dt / KNOT_MS > p.max_speed_kn)


def with_jump_flag(df: pl.DataFrame, p: CleaningParams) -> pl.DataFrame:
    """Add `jump`: the move from the previous fix of the same MMSI is physically impossible.

    Requires df sorted by (mmsi, ts).
    """
    c = pl.col
    same_prev = c("mmsi") == c("mmsi").shift(1)
    jump = _jump_between(c("lat").shift(1), c("lon").shift(1), c("ts").shift(1),
                         c("lat"), c("lon"), c("ts"), p)
    return df.with_columns((same_prev & jump).fill_null(False).alias("jump"))


def remove_spikes(df: pl.DataFrame, p: CleaningParams) -> tuple[pl.DataFrame, int]:
    """Iteratively drop isolated outliers: A -> X -> B where A->X and X->B are impossible
    but A->B is fine. A genuine relocation (A -> X -> X' -> ...) is left alone.

    Requires df sorted by (mmsi, ts). Returns (cleaned df, number removed).
    """
    c = pl.col
    removed = 0
    for _ in range(p.spike_passes):
        df = with_jump_flag(df, p)
        same_next = c("mmsi") == c("mmsi").shift(-1)
        skip_jump = _jump_between(c("lat").shift(1), c("lon").shift(1), c("ts").shift(1),
                                  c("lat").shift(-1), c("lon").shift(-1), c("ts").shift(-1), p)
        spike = (c("jump") & c("jump").shift(-1) & same_next & ~skip_jump).fill_null(False)
        mask = df.select(spike).to_series()
        n = int(mask.sum())
        if n == 0:
            break
        removed += n
        df = df.filter(~mask)
    return with_jump_flag(df, p), removed


# ---------------------------------------------------------------- day orchestration

def clean_day(stage_path: Path, positions_out: Path, static_out: Path,
              p: CleaningParams, bbox: BBox) -> dict:
    """Clean one staged day. Writes positions and static Parquet, returns step counts."""
    lf = pl.scan_parquet(stage_path)
    vessel, mmsi_ok, pos_ok, aoi = is_vessel(p), valid_mmsi(p), valid_position(), in_bbox(bbox)

    # One pass over the day to count what each rule rejects (funnel for RESULTS.md)
    counts = lf.select(
        n_raw=pl.len(),
        n_not_vessel=(~vessel).sum(),
        n_bad_mmsi=(vessel & ~mmsi_ok).sum(),
        n_bad_position=(vessel & mmsi_ok & ~pos_ok).sum(),
        n_outside_aoi=(vessel & mmsi_ok & pos_ok & ~aoi).sum(),
    ).collect(engine="streaming").row(0, named=True)

    keep = vessel & mmsi_ok & pos_ok & aoi
    df = (
        lf.filter(keep)
        .select(POSITION_COLS)
        .with_columns(normalize_sentinels(p))
        .collect(engine="streaming")
    )
    n_valid = df.height

    df = df.unique(subset=["mmsi", "ts", "lat", "lon"], keep="first").sort(["mmsi", "ts"])
    n_dedup = df.height

    df, n_low_res = drop_low_resolution(df, p)
    df, n_stale = drop_stale_repeats(df, p)
    df, n_spikes = remove_spikes(df, p)
    n_jumps = int(df["jump"].sum())

    positions_out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(positions_out)

    static = extract_static(lf.filter(vessel & mmsi_ok))
    static_out.parent.mkdir(parents=True, exist_ok=True)
    static.write_parquet(static_out)

    return {
        **counts,
        "n_duplicates": n_valid - n_dedup,
        "n_low_res": n_low_res,
        "n_stale_repeats": n_stale,
        "n_spikes": n_spikes,
        "n_clean": df.height,
        "n_jumps_flagged": n_jumps,
        "n_vessels": df["mmsi"].n_unique(),
        "n_static_records": static.height,
    }


def extract_static(lf: pl.LazyFrame) -> pl.DataFrame:
    """Distinct static/voyage records per MMSI with first/last time seen.

    A change of destination or ETA during the day produces a new record, which
    phase A5 uses to reconstruct voyages.
    """
    has_static = pl.any_horizontal(pl.col(c).is_not_null() for c in STATIC_COLS)
    return (
        lf.filter(has_static)
        .group_by(["mmsi", *STATIC_COLS])
        .agg(first_seen=pl.col("ts").min(), last_seen=pl.col("ts").max(), n_msgs=pl.len())
        .sort(["mmsi", "first_seen"])
        .collect(engine="streaming")
    )
