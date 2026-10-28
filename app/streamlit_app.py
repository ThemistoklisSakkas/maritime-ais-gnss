"""Part C - Streamlit dashboard: Maritime AIS Intelligence & GNSS Resilience.

Reads only saved outputs (outputs/dashboard, outputs/tables, outputs/figures), so it opens
instantly from a fresh clone without re-running the pipelines:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "outputs" / "dashboard"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"
VIEW = pdk.ViewState(latitude=56.0, longitude=10.5, zoom=5.2)

st.set_page_config(page_title="Maritime AIS & GNSS Resilience", page_icon="⚓", layout="wide")


# ---------------------------------------------------------------- loaders

@st.cache_data
def table(name: str) -> pl.DataFrame:
    return pl.read_parquet(DASH / f"{name}.parquet")


@st.cache_data
def summary(name: str) -> dict:
    p = TABLES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def figure(name: str, caption: str = "") -> None:
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def deck(layers: list, tooltip: dict | None = None) -> None:
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=VIEW, map_provider="carto",
                             map_style=pdk.map_styles.CARTO_LIGHT, tooltip=tooltip))


def to_pd(df: pl.DataFrame) -> pd.DataFrame:
    """pydeck/Streamlit charts want pandas; timestamps become strings for tooltips."""
    return df.with_columns(pl.col(pl.Datetime).dt.strftime("%Y-%m-%d %H:%M")).to_pandas()


# ---------------------------------------------------------------- pages

def page_overview() -> None:
    st.title("⚓ Maritime AIS Intelligence & GNSS Resilience")
    st.markdown(
        "Two halves of one problem: **what AIS reveals about vessel behaviour and GNSS "
        "interference** (real Danish Maritime Authority data, 2–6 June 2025), and **how a "
        "ship's navigation can survive GNSS spoofing and jamming** (INS/GNSS simulation).")
    funnel = pl.read_csv(TABLES / "a1_cleaning_funnel.csv")
    a2, a4, a5, b5 = summary("a2_summary"), summary("a4_summary"), summary("a5_summary"), \
        summary("b5_summary")
    fa = b5.get("false_alarms", {})
    sweeps = {(r["attack"], r["grade"], r["level"]): r for r in b5.get("sweeps", [])}
    jump = sweeps.get(("jump", "tactical", 180.0), {})

    def card(col, label, value, note):
        col.metric(label, value)
        col.caption(note)

    st.subheader("Part A — AIS analytics (real data)")
    c = st.columns(4)
    card(c[0], "AIS messages", f"{funnel['n_raw'].sum() / 1e6:.1f} M",
         f"{funnel['n_clean'].sum() / 1e6:.1f} M kept after cleaning")
    card(c[1], "Dark gaps", f"{a2.get('dark_gaps', 0):,}",
         f"unexplained AIS silences ≥ 2 h; {a2.get('dark_gaps_moved_while_dark', 0)} "
         "moved while dark")
    card(c[2], "Interference", f"{a4.get('cell_hours_flagged', '–')} cell-h",
         "flagged GNSS interference, all around Bornholm")
    card(c[3], "ETA error", f"{a5.get('mae_model_h', 0):.1f} h",
         f"gradient boosting, {a5.get('model_improvement_pct', 0):.0f} % below dead "
         "reckoning")
    st.subheader("Part B — GNSS-resilient navigation (simulation)")
    c = st.columns(4)
    card(c[0], "False alarms", str(fa.get("tactical", {}).get("declarations", "–")),
         "in 20 h without attack (tactical IMU)")
    card(c[1], "Jump detection", "≤ 2 s",
         f"180 m spoof: {jump.get('err_end_protected_m', '–')} m error instead of 180 m")
    card(c[2], "Drift, tactical", "0.3 m/s",
         "slowest spoofed drift detected in 100 % of runs (tactical MEMS IMU)")
    card(c[3], "Drift, nav-grade", "0.2 m/s",
         "same with a navigation-grade IMU: better IMU, slower spoofing caught")
    st.divider()
    left, right = st.columns(2)
    with left:
        figure("a1_density.png", "Cleaned AIS fixes, one day")
    with right:
        figure("a4_interference_timeseries.png", "GNSS interference indicators over time")
    st.caption("Numbers come from outputs/tables; methods and caveats in README.md, "
               "RESULTS.md and LIMITATIONS.md.")


def page_dark() -> None:
    st.header("Dark activity: AIS silences of 2 h or more that nothing explains")
    st.markdown("Gaps in ports/anchorages, in areas without receiver coverage and position "
                "jumps were explained away first (`a2_gap_categories.png`). What remains is "
                "shown here. **Unexplained is not guilty**: a transponder fault looks the same.")
    g = table("dark_gaps").with_columns(pl.col("ship_type").fill_null("unknown"))
    f1, f2, f3 = st.columns(3)
    types = sorted(g["ship_type"].unique().to_list())
    default = [t for t in ("Cargo", "Tanker", "Passenger", "Fishing") if t in types]
    sel = f1.multiselect("Ship type", types, default=default)
    moved = f2.checkbox("Only vessels that moved ≥ 1 NM while dark", value=True)
    min_h = f3.slider("Minimum gap [h]", 2.0, 48.0, 2.0, 0.5)
    v = g.filter(pl.col("ship_type").is_in(sel) & (pl.col("gap_h") >= min_h))
    if moved:
        v = v.filter(pl.col("moved_while_dark"))
    st.write(f"**{v.height}** gaps from **{v['mmsi'].n_unique()}** vessels")
    d = to_pd(v.with_columns(pl.col("gap_h").round(1), pl.col("gap_dist_nm").round(1)))
    deck([
        pdk.Layer("LineLayer", d, get_source_position=["start_lon", "start_lat"],
                  get_target_position=["end_lon", "end_lat"], get_color=[200, 30, 30, 160],
                  get_width=2, pickable=True),
        pdk.Layer("ScatterplotLayer", d, get_position=["start_lon", "start_lat"],
                  get_fill_color=[200, 30, 30, 200], get_radius=900, radius_min_pixels=4,
                  pickable=True),
    ], {"text": "{name} ({ship_type})\nsilent {gap_h} h, moved {gap_dist_nm} NM\n"
                "from {gap_start}"})
    st.dataframe(d.sort_values("gap_h", ascending=False), use_container_width=True,
                 hide_index=True)


def page_sts() -> None:
    st.header("Ship-to-ship encounters")
    st.markdown("Two vessels < 500 m apart, both < 3 kn, for ≥ 1 h, outside ports and "
                "anchorages (KD-tree pairing per 5-min bin). Encounters right next to the "
                "coastline are usually berths in small harbours missing from the port list.")
    e = table("encounters")
    f1, f2, f3 = st.columns(3)
    classes = sorted(e["pair_class"].unique().to_list())
    sel = f1.multiselect("Pair class", classes,
                         default=[c for c in classes if c.startswith("cargo_tanker")])
    offshore = f2.checkbox("Only offshore (> 1 km from land)", value=True)
    dark = f3.checkbox("Only with a dark gap within 12 h", value=False)
    v = e.filter(pl.col("pair_class").is_in(sel))
    if offshore:
        v = v.filter(~pl.col("near_coast"))
    if dark:
        v = v.filter(pl.col("dark_nearby"))
    st.write(f"**{v.height}** encounters")
    d = to_pd(v.with_columns(pl.col("duration_h").round(1), pl.col("mean_dist_m").round(0),
                             radius=pl.col("duration_h") * 400 + 800))
    deck([pdk.Layer("ScatterplotLayer", d, get_position=["lon", "lat"], get_radius="radius",
                    radius_min_pixels=4, get_fill_color=[30, 90, 200, 170], pickable=True)],
         {"text": "{name_a} ({ship_type_a}) + {name_b} ({ship_type_b})\n"
                  "{duration_h} h at {mean_dist_m} m, from {start}"})
    st.dataframe(d.drop(columns=["radius"]).sort_values("duration_h", ascending=False),
                 use_container_width=True, hide_index=True)
    figure("a3_encounter_example.png",
           "Two tankers alongside for 9 h at the Skagen anchorage: consistent with bunkering")


def page_gnss() -> None:
    st.header("GNSS interference indicators in AIS")
    st.markdown("Per 0.5° cell and hour: distinct vessels with physically impossible position "
                "jumps or fixes deep inland after a jump. Flagged when ≥ 3 vessels and ≥ 5 % "
                "of those present are affected. Jump destinations never converged, so the "
                "pattern is consistent with **jamming**, not coordinated spoofing.")
    cells = table("interference_cells").with_columns(day=pl.col("hour").dt.date(),
                                                     hh=pl.col("hour").dt.hour())
    days = sorted(cells["day"].unique().to_list())
    f1, f2 = st.columns(2)
    day = f1.selectbox("Day", days, index=days.index(max(
        days, key=lambda d: cells.filter((pl.col("day") == d) & pl.col("flagged")).height)))
    hour = f2.slider("Hour (UTC)", 0, 23, 15)
    v = cells.filter((pl.col("day") == day) & (pl.col("hh") == hour) & (pl.col("affected") > 0))
    half = 0.25
    d = to_pd(v.select("lat", "lon", "n", "affected", "flagged",
                       share_pct=(pl.col("share") * 100).round(1),
                       alpha=(pl.col("share").clip(0, 0.4) / 0.4 * 200 + 40).cast(pl.Int32)))
    # pydeck serialises plain Python lists, not the numpy arrays polars/pandas would give
    d["polygon"] = [[[lo - half, la - half], [lo + half, la - half], [lo + half, la + half],
                     [lo - half, la + half]] for lo, la in zip(d["lon"], d["lat"], strict=True)]
    d["fill"] = [[220, 60, 20, int(a)] for a in d["alpha"]]
    d["line"] = [[0, 0, 0, 255] if f else [120, 120, 120, 120] for f in d["flagged"]]
    st.write(f"{v.height} cells with affected vessels, "
             f"**{int(v['flagged'].sum())} flagged** (black outline) at {day} {hour:02d}:00 UTC")
    deck([pdk.Layer("PolygonLayer", d, get_polygon="polygon", get_fill_color="fill",
                    get_line_color="line", line_width_min_pixels=2, stroked=True,
                    pickable=True)],
         {"text": "{affected} of {n} vessels affected ({share_pct} %)\nflagged: {flagged}"})
    ts = pl.read_csv(TABLES / "a4_hourly_bornholm_vs_rest.csv", try_parse_dates=True)
    st.subheader("Affected vessels per hour (flagged cell-hours)")
    st.line_chart(ts.select("hour", pl.col("affected_region").alias("Bornholm box"),
                            pl.col("affected_rest").alias("rest of AOI")).to_pandas(),
                  x="hour")
    figure("a4_interference_daily.png")


def page_eta() -> None:
    st.header("ETA prediction")
    st.markdown("Remaining time to arrival predicted three ways on held-out later arrivals: "
                "**dead reckoning** (great-circle distance / SOG), **gradient boosting**, and "
                "the **AIS-reported ETA** typed in by the crew (when its destination matches). "
                "The ECDIS route-based ETA is not transmitted by AIS.")
    eta = table("eta_test")
    voyages = (eta.group_by("arrival_id")
                  .agg(pl.first("port_name"), pl.first("mmsi"), pl.max("target_h"),
                       pl.col("crew_h").is_not_null().any().alias("has_crew"))
                  .sort("target_h", descending=True))
    labels = {r["arrival_id"]: f"{r['port_name']} - MMSI {r['mmsi']} - "
                               f"{r['target_h']:.0f} h run-in"
                               + (" - with AIS ETA" if r["has_crew"] else "")
              for r in voyages.iter_rows(named=True)}
    aid = st.selectbox("Voyage", list(labels), format_func=labels.get)
    v = eta.filter(pl.col("arrival_id") == aid).sort("t")
    left, right = st.columns([3, 2])
    with left:
        st.line_chart(v.select(pl.col("t").alias("time"),
                               pl.col("target_h").alias("truth"),
                               pl.col("baseline_h").alias("dead reckoning"),
                               pl.col("model_h").alias("gradient boosting"),
                               pl.col("crew_h").alias("AIS-reported ETA")).to_pandas(),
                      x="time", y_label="hours to arrival")
    with right:
        d = to_pd(v)
        port = d.iloc[[0]]
        deck([pdk.Layer("ScatterplotLayer", d, get_position=["lon", "lat"], get_radius=600,
                        radius_min_pixels=2, get_fill_color=[30, 90, 200, 180]),
              pdk.Layer("ScatterplotLayer", port, get_position=["port_lon", "port_lat"],
                        get_radius=3000, radius_min_pixels=6,
                        get_fill_color=[220, 40, 40, 200])])
    mae = pl.read_csv(TABLES / "a5_eta_mae_by_horizon.csv")
    st.subheader("MAE on the test set [h]")
    st.dataframe(mae.select("horizon", "n", "mae_baseline_h", "mae_model_h",
                            "plausible_mae_crew_h", "crew_median_ae_crew_h")
                    .with_columns(pl.col(pl.Float64).round(2)).to_pandas(),
                 hide_index=True, use_container_width=True)
    figure("a5_eta_evaluation.png")


def page_nav() -> None:
    st.header("GNSS-resilient navigation (simulation)")
    st.markdown("2D INS/GNSS error-state EKF (8 states) on a simulated 1-h coastal passage, "
                "attacked by GNSS spoofing (jump, slow drift) and jamming. The integrity "
                "monitor combines a χ² gate with a **backup-INS window test**, rolls back to "
                "the clean backup and coasts on INS + speed log.")
    b5 = summary("b5_summary")
    rows = pl.DataFrame(b5.get("sweeps", []))
    if rows.height:
        f1, f2, f3 = st.columns(3)
        attack = f1.selectbox("Attack", ["drift", "jump"])
        grades = sorted(rows.filter(pl.col("attack") == attack)["grade"].unique().to_list())
        grade = f2.selectbox("IMU grade", grades, index=grades.index("tactical")
                             if "tactical" in grades else 0)
        levels = sorted(rows.filter((pl.col("attack") == attack) & (pl.col("grade") == grade))
                        ["level"].to_list())
        unit = "m/s" if attack == "drift" else "m"
        level = f3.select_slider(f"Size [{unit}]", levels, value=levels[len(levels) // 2])
        r = rows.filter((pl.col("attack") == attack) & (pl.col("grade") == grade)
                        & (pl.col("level") == level)).row(0, named=True)
        c = st.columns(4)
        c[0].metric("Detected", f"{100 * r['p_detect']:.0f} %")
        c[0].caption(f"during the attack, {r['runs']} Monte Carlo runs")
        c[1].metric("Delay", "–" if r["delay_median_s"] is None
                    else f"{r['delay_median_s']:.0f} s")
        c[1].caption("median time to detection")
        c[2].metric("Unprotected", f"{r['err_end_unprotected_m']:.0f} m")
        c[2].caption("median error at end of the 15-min attack")
        c[3].metric("Protected", f"{r['err_end_protected_m']:.0f} m")
        c[3].caption("same runs, with the integrity monitor")
    tabs = st.tabs(["Detection curves", "Protected vs unprotected", "Attacks (no protection)",
                    "Filter consistency", "Sensors"])
    with tabs[0]:
        figure("b5_detection_curves.png")
        figure("b5_error_reduction.png")
    with tabs[1]:
        figure("b4_protected_vs_unprotected.png")
        figure("b4_window_test_calibration.png")
    with tabs[2]:
        figure("b3_attacks_unprotected.png")
        figure("b3_drift_stealth.png")
    with tabs[3]:
        figure("b2_single_run.png")
        figure("b2_consistency.png")
    with tabs[4]:
        figure("b1_scenario.png")
        figure("b1_free_inertial_drift.png")
        figure("b1_allan_deviation.png")


PAGES = {
    "Overview": page_overview,
    "Dark activity": page_dark,
    "Ship-to-ship encounters": page_sts,
    "GNSS interference": page_gnss,
    "ETA prediction": page_eta,
    "INS/GNSS resilience": page_nav,
}

choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.markdown("---\nData: Danish Maritime Authority AIS, World Port Index, Natural "
                    "Earth. Simulation for Part B.\n\nSee README.md for methods and "
                    "LIMITATIONS.md for caveats.")
PAGES[choice]()
