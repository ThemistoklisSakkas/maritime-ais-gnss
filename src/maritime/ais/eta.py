"""A5 - ETA prediction to the port of arrival.

1. Arrivals: a vessel enters a WPI port zone and STAYS there (>= `min_stay_h`, mostly
   below `stop_sog_kn`). Merely passing a port (Helsingør in the Øresund) is not an arrival.
   Arrival time = first fix inside the zone.
2. Samples: every `sample_every_min` while the vessel was at sea on its way in (after its
   previous port stay, up to `max_horizon_h` before arrival). Target = hours to arrival.
3. Predictors on identical samples:
     baseline  great-circle distance / recent mean SOG ("dead reckoning by hand")
     model     gradient boosting on distance, speeds, heading-to-port, position, port, ship;
               trained on log(1 + hours) so a 30-min miss counts as much at 2 h as a 3-h
               miss at 20 h (chosen on a validation split inside the training period)
     crew      the AIS-reported ETA (msg 5 voyage data), when its destination matches the port.
               This is NOT the ECDIS/GNSS "ETA to waypoint", which the bridge computes
               continuously from route distance-to-go / SOG and which AIS does not carry. The
               AIS field is entered manually by the OOW and updated "as necessary", so it
               reflects the commercial voyage plan (pilot, berth window) and can go stale.
               Code names use "crew_" for brevity.
     The baseline is effectively the bridge's dead-reckoning ETA, but with great-circle
     instead of route distance, since the passage plan is unknown.
4. Evaluation on a TIME split: train on arrivals before a cutoff, test on later arrivals, so
   no voyage contributes to both sides and the model is scored on the "future".

The destination is assumed known (the port actually reached), which is the situation of a
port authority or charterer; inferring the destination itself is a different problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from maritime.geo import angle_diff_expr, haversine_expr, initial_bearing_deg
from maritime.zones import nearest_port_gridded


@dataclass(frozen=True)
class ETAParams:
    ship_types: tuple[str, ...]
    resample_min: int
    min_stay_h: float
    stop_sog_kn: float
    min_stop_share: float
    sample_every_min: int
    max_horizon_h: float
    state_tolerance_min: int
    sog_floor_kn: float
    train_share: float
    zone_cell_deg: float
    crew_plausible_max_h: float

    @classmethod
    def from_dict(cls, d: dict) -> ETAParams:
        return cls(tuple(d["ship_types"]), int(d["resample_min"]), float(d["min_stay_h"]),
                   float(d["stop_sog_kn"]), float(d["min_stop_share"]),
                   int(d["sample_every_min"]), float(d["max_horizon_h"]),
                   int(d["state_tolerance_min"]), float(d["sog_floor_kn"]),
                   float(d["train_share"]), float(d["zone_cell_deg"]),
                   float(d["crew_plausible_max_h"]))


FEATURES = ["dist_nm", "sog_kn", "sog_mean_1h", "baseline_h", "off_course_deg",
            "lat", "lon", "port_lat", "port_lon", "length", "is_tanker"]


# ---------------------------------------------------------------- tracks and arrivals

def minute_tracks(points_paths: list[Path], mmsis: pl.Series, ports: pl.DataFrame,
                  p: ETAParams) -> pl.DataFrame:
    """One fix per vessel per `resample_min`, with the port zone index (-1 = at sea) and a
    1-hour trailing mean SOG. Restricted to the vessels in `mmsis`."""
    parts = []
    for f in points_paths:
        parts.append(
            pl.scan_parquet(f).filter(pl.col("mmsi").is_in(mmsis.implode()))
            .select("mmsi", "ts", "lat", "lon", "sog", "cog")
            .with_columns(pl.col("ts").dt.truncate(f"{p.resample_min}m"))
            .group_by(["mmsi", "ts"]).agg(pl.all().last())
            .collect()
        )
    df = pl.concat(parts).sort(["mmsi", "ts"])
    df = df.with_columns(port=pl.Series(nearest_port_gridded(df, ports, p.zone_cell_deg)))
    return df.with_columns(
        sog_mean_1h=pl.col("sog").rolling_mean_by("ts", window_size="1h").over("mmsi"))


def find_stays(tracks: pl.DataFrame, p: ETAParams) -> pl.DataFrame:
    """Consecutive runs of fixes in the same port zone that qualify as a port stay."""
    c = pl.col
    run = ((c("mmsi") != c("mmsi").shift(1)) | (c("port") != c("port").shift(1))) \
        .fill_null(True).cum_sum()
    first_fix = c("ts").min().over("mmsi")
    runs = (
        tracks.with_columns(run=run, vessel_first=first_fix)
        .filter(c("port") >= 0)
        .group_by("run")
        .agg(c("mmsi").first(), c("port").first(), start=c("ts").min(), end=c("ts").max(),
             stop_share=(c("sog") < p.stop_sog_kn).mean(),
             starts_at_first_fix=(c("ts").min() == c("vessel_first").first()))
        .with_columns(stay_h=(c("end") - c("start")).dt.total_seconds() / 3600)
    )
    return (runs.filter((c("stay_h") >= p.min_stay_h) & (c("stop_share") >= p.min_stop_share))
                .drop("run").sort(["mmsi", "start"]))


def arrivals_from_stays(stays: pl.DataFrame, tracks: pl.DataFrame) -> pl.DataFrame:
    """Arrival = start of a stay. The trip began at the end of the previous stay (or at the
    vessel's first fix). Stays already in progress when the data starts are not arrivals."""
    c = pl.col
    first = tracks.group_by("mmsi").agg(first_fix=c("ts").min())
    return (
        stays.join(first, on="mmsi")
        .with_columns(trip_start=c("end").shift(1).over("mmsi").fill_null(c("first_fix")))
        .filter(~c("starts_at_first_fix"))
        .select("mmsi", "port", c("start").alias("arrival"), "trip_start")
    )


# ---------------------------------------------------------------- samples and features

def build_samples(arrivals: pl.DataFrame, tracks: pl.DataFrame, ports: pl.DataFrame,
                  registry: pl.DataFrame, p: ETAParams) -> pl.DataFrame:
    """Vessel state every `sample_every_min` on the way in, with features and target."""
    c = pl.col
    every = f"{p.sample_every_min}m"
    grid = (
        arrivals.with_row_index("arrival_id")
        .with_columns(t=pl.datetime_ranges(c("trip_start").dt.truncate(every) + timedelta(
            minutes=p.sample_every_min), c("arrival") - timedelta(minutes=1), every))
        .explode("t").drop_nulls("t")
        .with_columns(target_h=(c("arrival") - c("t")).dt.total_seconds() / 3600)
        .filter(c("target_h") <= p.max_horizon_h)
        .sort("t")
    )
    state = tracks.select("mmsi", c("ts").alias("t_state"), "lat", "lon", "sog", "cog",
                          "sog_mean_1h", c("port").alias("port_now")).sort("t_state")
    s = grid.join_asof(state, left_on="t", right_on="t_state", by="mmsi", strategy="backward",
                       tolerance=f"{p.state_tolerance_min}m", check_sortedness=False)
    s = s.drop_nulls(["lat", "sog"]).filter(c("port_now") < 0)  # observed, and at sea

    port_xy = ports.with_row_index("pid").select(
        c("pid").cast(pl.Int64).alias("port"), port_lat=c("lat"), port_lon=c("lon"))
    s = s.join(port_xy, on="port", how="left")
    bearing = initial_bearing_deg(s["lat"].to_numpy(), s["lon"].to_numpy(),
                                  s["port_lat"].to_numpy(), s["port_lon"].to_numpy())
    reg = registry.select("mmsi", "length", is_tanker=(c("ship_type") == "Tanker").cast(pl.Int8))
    s = (
        s.with_columns(bearing_to_port=pl.Series(bearing))
        .with_columns(
            dist_nm=haversine_expr(c("lat"), c("lon"), c("port_lat"), c("port_lon")) / 1852,
            off_course_deg=angle_diff_expr(c("cog"), c("bearing_to_port")).abs(),
        )
        .with_columns(baseline_h=c("dist_nm") / pl.max_horizontal(
            c("sog_mean_1h").fill_null(c("sog")), pl.lit(p.sog_floor_kn)))
        .join(reg, on="mmsi", how="left")
        .rename({"sog": "sog_kn"})
    )
    return s.select("arrival_id", "mmsi", "port", "arrival", "t", "target_h", *FEATURES)


# ---------------------------------------------------------------- crew ETA

_LOCODE = re.compile(r"[^A-Z]")


def normalize_destination(s: str | None) -> str:
    """'SE GOT' / 'se-got' -> 'SEGOT'; 'GOTHENBURG ' -> 'GOTHENBURG'."""
    return _LOCODE.sub("", s.upper()) if s else ""


def destination_matches(dest: str | None, locode: str | None, port_name: str | None) -> bool:
    d = normalize_destination(dest)
    if not d:
        return False
    loc = normalize_destination(locode)
    name = normalize_destination(port_name)
    if loc and loc in d:
        return True
    # name match only for strings long enough not to match by accident ("DK" in "DKTEST")
    return len(name) >= 4 and len(d) >= 4 and (name in d or d in name)


def parse_crew_eta(eta: pl.Expr, reported: pl.Expr) -> pl.Expr:
    """AIS carries ETA as month/day/hour/minute only; DMA prepends a year that is sometimes
    wrong (e.g. 2026 in June 2025). Re-attach the year that puts the ETA closest to the
    time it was reported."""
    md = eta.str.slice(0, 5)                       # "dd/mm"
    hm = eta.str.slice(11, 5)                      # "HH:MM"
    y = reported.dt.year()
    cands = [pl.concat_str([md, pl.lit("/"), (y + k).cast(pl.String), pl.lit(" "), hm])
             .str.strptime(pl.Datetime("us"), "%d/%m/%Y %H:%M", strict=False)
             .dt.replace_time_zone("UTC") for k in (-1, 0, 1)]
    best = cands[1]
    for cand in (cands[0], cands[2]):
        closer = (cand - reported).abs() < (best - reported).abs()
        best = pl.when(cand.is_not_null() & (best.is_null() | closer)).then(cand).otherwise(best)
    return best


def attach_crew_eta(samples: pl.DataFrame, static: pl.DataFrame, ports: pl.DataFrame
                    ) -> pl.DataFrame:
    """Latest crew-reported (destination, ETA) at each sample time; keep the ETA only when
    the destination names the port the vessel actually reached."""
    c = pl.col
    st = (static.select("mmsi", "destination", "eta", reported=c("first_seen"))
                .drop_nulls(["eta"])
                .with_columns(crew_eta=parse_crew_eta(c("eta"), c("reported")))
                .drop_nulls("crew_eta").sort("reported"))
    j = samples.sort("t").join_asof(st, left_on="t", right_on="reported", by="mmsi",
                                    strategy="backward", check_sortedness=False)
    names = ports.with_row_index("pid").select(c("pid").cast(pl.Int64).alias("port"),
                                               "locode", c("port").alias("port_name"))
    j = j.join(names, on="port", how="left")
    match = [destination_matches(d, lc, pn) for d, lc, pn in
             zip(j["destination"].to_list(), j["locode"].to_list(), j["port_name"].to_list(),
                 strict=True)]
    return j.with_columns(dest_match=pl.Series(match, dtype=pl.Boolean)).with_columns(
        crew_h=pl.when(c("dest_match"))
        .then((c("crew_eta") - c("t")).dt.total_seconds() / 3600)
    ).drop("locode", "port_name", "reported")


# ---------------------------------------------------------------- model and evaluation

def time_split(samples: pl.DataFrame, train_share: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split by ARRIVAL time: whole voyages go to one side; test = the latest arrivals."""
    arr = samples.select("arrival_id", "arrival").unique().sort("arrival")
    cutoff = arr["arrival"][int(len(arr) * train_share)]
    return (samples.filter(pl.col("arrival") < cutoff),
            samples.filter(pl.col("arrival") >= cutoff))


def fit_model(train: pl.DataFrame, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400,
                                          learning_rate=0.05, max_leaf_nodes=31,
                                          random_state=seed)
    model.fit(train.select(FEATURES).to_numpy(), np.log1p(train["target_h"].to_numpy()))
    return model


def predict_hours(model, df: pl.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(df.select(FEATURES).to_numpy()))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    ok = ~np.isnan(pred)
    return float(np.mean(np.abs(pred[ok] - truth[ok]))) if ok.any() else float("nan")


HORIZON_BINS = [0, 3, 6, 12, 24, 48]


def evaluate(test: pl.DataFrame, crew_plausible_max_h: float) -> pl.DataFrame:
    """MAE in hours per horizon bin for baseline and model, plus crew-ETA comparisons on
    like-for-like subsets: every matched crew ETA, and only "plausible" ones (not already in
    the past, not more than `crew_plausible_max_h` ahead), since stale ETAs are common."""
    c = pl.col
    edges = list(zip(HORIZON_BINS[:-1], HORIZON_BINS[1:], strict=True))
    labels = [f"{a}-{b} h" for a, b in edges]
    label = pl.lit(None, pl.String)
    for (a, b), lab in zip(reversed(edges), reversed(labels), strict=True):
        label = pl.when((c("target_h") > a) & (c("target_h") <= b)).then(pl.lit(lab)) \
                  .otherwise(label)
    t = test.with_columns(bin=label)

    def maes(d: pl.DataFrame, prefix: str) -> dict:
        y = d["target_h"].to_numpy()
        out = {f"{prefix}n": d.height,
               f"{prefix}mae_baseline_h": mae(d["baseline_h"].to_numpy(), y),
               f"{prefix}mae_model_h": mae(d["model_h"].to_numpy(), y)}
        if prefix:
            err = np.abs(d["crew_h"].to_numpy() - y)
            out[f"{prefix}mae_crew_h"] = float(err.mean()) if err.size else float("nan")
            out[f"{prefix}median_ae_crew_h"] = float(np.median(err)) if err.size else float("nan")
        return out

    rows = []
    for name, sub_df in [("all", t), *[(lab, t.filter(c("bin") == lab)) for lab in labels]]:
        crew = sub_df.filter(c("crew_h").is_not_null())
        plaus = crew.filter(c("crew_h").is_between(0, crew_plausible_max_h))
        rows.append({"horizon": name, **maes(sub_df, ""), **maes(crew, "crew_"),
                     **maes(plaus, "plausible_")})
    return pl.DataFrame(rows)
