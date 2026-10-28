"""Static figures for the README (matplotlib, saved as PNG)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from maritime.config import BBox


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def dark_gap_map(gaps: pl.DataFrame, coverage: pl.DataFrame, cell_deg: float, bbox: BBox,
                 out_png: Path, focus_types: list[str]) -> Path:
    """Dark gaps drawn as start -> end lines over the empirical coverage map."""
    from matplotlib.colors import LogNorm

    plt = _plt()
    lon_min, lat_min, lon_max, lat_max = bbox
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(_coverage_grid(coverage, cell_deg, bbox), origin="lower",
              extent=[lon_min, lon_max, lat_min, lat_max], cmap="Greys", norm=LogNorm(),
              alpha=0.6, aspect="auto")
    focus = pl.col("ship_type").is_in(focus_types).fill_null(False)
    moved = pl.col("moved_while_dark")
    layers = (
        (gaps.filter(~moved), "o", 10, "none", "tab:gray", "stayed in place (any type)"),
        (gaps.filter(moved & ~focus), "o", 12, "tab:blue", "tab:blue", "moved, other / unknown"),
        (gaps.filter(moved & focus), "D", 30, "tab:red", "tab:red",
         f"moved, {' / '.join(focus_types)}"),
    )
    for sub, marker, size, face, edge, label in layers:
        if face != "none":
            for r in sub.iter_rows(named=True):
                ax.annotate("", xy=(r["end_lon"], r["end_lat"]),
                            xytext=(r["start_lon"], r["start_lat"]),
                            arrowprops={"arrowstyle": "->", "color": edge, "lw": 0.9})
        ax.scatter(sub["start_lon"], sub["start_lat"], s=size, marker=marker,
                   facecolors=face, edgecolors=edge, label=f"{label} (n={sub.height})")
    ax.set_aspect(1 / np.cos(np.radians((lat_min + lat_max) / 2)))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Dark AIS gaps: marker = last fix before silence, arrow = to first fix after\n"
                 "grey = vessel-hours received per cell (empirical coverage)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def _coverage_grid(coverage: pl.DataFrame, cell_deg: float, bbox: BBox) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = bbox
    cov = coverage.group_by(["cx", "cy"]).agg(pl.col("n").sum())
    nx = int(round((lon_max - lon_min) / cell_deg))
    ny = int(round((lat_max - lat_min) / cell_deg))
    grid = np.full((ny, nx), np.nan)
    ix = cov["cx"].to_numpy() - int(round(lon_min / cell_deg))
    iy = cov["cy"].to_numpy() - int(round(lat_min / cell_deg))
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    grid[iy[ok], ix[ok]] = cov["n"].to_numpy()[ok]
    return grid


def encounter_map(ep: pl.DataFrame, coverage: pl.DataFrame, cell_deg: float, bbox: BBox,
                  out_png: Path) -> Path:
    """STS candidates over the coverage map; marker size = duration, ring = dark nearby."""
    from matplotlib.colors import LogNorm

    plt = _plt()
    lon_min, lat_min, lon_max, lat_max = bbox
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(_coverage_grid(coverage, cell_deg, bbox), origin="lower",
              extent=[lon_min, lon_max, lat_min, lat_max], cmap="Greys", norm=LogNorm(),
              alpha=0.5, aspect="auto")
    styles = {"cargo_tanker_pair": "tab:red", "cargo_tanker_with_unknown": "tab:orange",
              "fishing_pair": "tab:green", "service_involved": "tab:gray", "other": "tab:blue"}
    for cls, color in styles.items():
        sub = ep.filter(pl.col("pair_class") == cls)
        if sub.is_empty():
            continue
        ax.scatter(sub["lon"], sub["lat"], s=10 + 12 * sub["duration_h"].to_numpy(),
                   color=color, alpha=0.7, edgecolors="none", label=f"{cls} (n={sub.height})")
    dark = ep.filter(pl.col("dark_nearby"))
    ax.scatter(dark["lon"], dark["lat"], s=30 + 12 * dark["duration_h"].to_numpy(),
               facecolors="none", edgecolors="black", lw=1.2,
               label=f"dark gap within 12 h (n={dark.height})")
    ax.set_aspect(1 / np.cos(np.radians((lat_min + lat_max) / 2)))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Ship-to-ship encounter candidates (< 500 m, both < 3 kn, >= 1 h, "
                 "outside ports/anchorages)\nmarker size = duration")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def encounter_detail(tracks: pl.DataFrame, enc: dict, out_png: Path) -> Path:
    """Two vessels' tracks around one encounter, plus their separation over time."""
    from maritime.geo import haversine_m

    plt = _plt()
    a, b = enc["mmsi_a"], enc["mmsi_b"]
    ta = tracks.filter(pl.col("mmsi") == a).sort("ts")
    tb = tracks.filter(pl.col("mmsi") == b).sort("ts")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for t, color, label in ((ta, "tab:red", f"{enc.get('name_a')} ({enc.get('ship_type_a')})"),
                            (tb, "tab:blue", f"{enc.get('name_b')} ({enc.get('ship_type_b')})")):
        ax[0].plot(t["lon"], t["lat"], "-", color=color, lw=1, label=label)
        ax[0].plot(t["lon"][0], t["lat"][0], "o", color=color, ms=5)
    ax[0].plot(enc["lon"], enc["lat"], "k*", ms=14, label="encounter centre")
    ax[0].set_aspect(1 / np.cos(np.radians(enc["lat"])))
    ax[0].set_xlabel("Longitude [deg]")
    ax[0].set_ylabel("Latitude [deg]")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Tracks (dot = start of window)")

    # separation on a common 1-minute grid (as-of join, nearest fix within 5 min)
    grid = pl.DataFrame({"ts": pl.datetime_range(max(ta["ts"].min(), tb["ts"].min()),
                                                 min(ta["ts"].max(), tb["ts"].max()),
                                                 "1m", eager=True)})
    tol = "5m"
    j = (grid.join_asof(ta.select("ts", "lat", "lon"), on="ts", strategy="nearest",
                        tolerance=tol)
         .join_asof(tb.select("ts", pl.col("lat").alias("lat_b"), pl.col("lon").alias("lon_b")),
                    on="ts", strategy="nearest", tolerance=tol).drop_nulls())
    d = haversine_m(j["lat"].to_numpy(), j["lon"].to_numpy(),
                    j["lat_b"].to_numpy(), j["lon_b"].to_numpy())
    ax[1].plot(j["ts"].to_numpy(), d, color="black", lw=1)
    ax[1].axhline(500, color="tab:red", ls="--", label="500 m threshold")
    ax[1].axvspan(enc["start"], enc["end"], color="tab:orange", alpha=0.2, label="encounter")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Separation [m] (log)")
    ax[1].set_xlabel("Time (UTC)")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"Separation, {enc['duration_h']:.1f} h encounter")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def interference_daily_maps(cells: pl.DataFrame, coverage: pl.DataFrame, cell_deg: float,
                            coverage_cell_deg: float, bbox: BBox, out_png: Path) -> Path:
    """One panel per day: affected vessel-hours in flagged cell-hours, over coverage."""
    from matplotlib.colors import LogNorm

    plt = _plt()
    lon_min, lat_min, lon_max, lat_max = bbox
    flagged = cells.filter(pl.col("flagged")).with_columns(day=pl.col("hour").dt.date())
    days = sorted(cells["hour"].dt.date().unique().to_list())
    bg = _coverage_grid(coverage, coverage_cell_deg, bbox)
    nx = int(round((lon_max - lon_min) / cell_deg))
    ny = int(round((lat_max - lat_min) / cell_deg))
    vmax = max(1, int(flagged.group_by(["day", "cx", "cy"]).agg(pl.col("affected").sum())
                      ["affected"].max() or 1))

    fig, axes = plt.subplots(1, len(days), figsize=(4.2 * len(days), 4.6), squeeze=False)
    im = None
    for ax, day in zip(axes[0], days, strict=True):
        ax.imshow(bg, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max],
                  cmap="Greys", norm=LogNorm(), alpha=0.35, aspect="auto")
        d = flagged.filter(pl.col("day") == day).group_by(["cx", "cy"]) \
                   .agg(pl.col("affected").sum())
        grid = np.full((ny, nx), np.nan)
        ix = d["cx"].to_numpy() - int(round(lon_min / cell_deg))
        iy = d["cy"].to_numpy() - int(round(lat_min / cell_deg))
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        grid[iy[ok], ix[ok]] = d["affected"].to_numpy()[ok]
        im = ax.imshow(grid, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max],
                       cmap="inferno_r", norm=LogNorm(vmin=1, vmax=vmax), aspect="auto")
        ax.set_aspect(1 / np.cos(np.radians((lat_min + lat_max) / 2)))
        ax.set_title(f"{day}  ({int(d['affected'].sum() or 0)} vessel-h)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8,
                 label=f"affected vessel-hours per {cell_deg}° cell (flagged only)")
    fig.suptitle("GNSS interference indicators in AIS, by day "
                 "(impossible jumps / fixes deep inland, >= 3 vessels and >= 5 % per cell-hour)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def interference_timeseries(hourly: pl.DataFrame, region_name: str, out_png: Path) -> Path:
    """Hourly affected vessels inside a region vs the rest of the AOI, plus regional share."""
    plt = _plt()
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for col, color, label in (("affected_region", "tab:red", region_name),
                              ("affected_rest", "tab:blue", "rest of AOI")):
        ax[0].plot(hourly["hour"].to_numpy(), hourly[col].to_numpy(), color=color, lw=1.2,
                   label=label)
    ax[0].set_ylabel("affected vessels\n(flagged cell-hours)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(hourly["hour"].to_numpy(), 100 * hourly["share_region"].to_numpy(),
                       color="tab:red", alpha=0.4, step="mid")
    ax[1].set_ylabel(f"% of vessels in\n{region_name} affected")
    ax[1].set_xlabel("Time (UTC)")
    ax[1].grid(alpha=0.3)
    fig.suptitle("GNSS interference indicators over time")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def eta_evaluation(ev: pl.DataFrame, test: pl.DataFrame, out_png: Path) -> Path:
    """Left: MAE by horizon for baseline/model (+ crew on its subset). Right: predicted vs true."""
    plt = _plt()
    bins = ev.filter(pl.col("horizon") != "all")
    x = np.arange(bins.height)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    w = 0.38
    ax[0].bar(x - w / 2, bins["mae_baseline_h"], w, color="#999",
              label="baseline: great-circle distance / SOG")
    ax[0].bar(x + w / 2, bins["mae_model_h"], w, color="tab:blue", label="gradient boosting")
    for col, style, label in (("crew_median_ae_crew_h", "o--",
                               "AIS-reported ETA (manual), median abs error"),
                              ("plausible_mae_crew_h", "s:",
                               "AIS-reported ETA (manual), MAE, plausible only")):
        crew = bins[col].to_numpy()
        ok = ~np.isnan(crew)
        ax[0].plot(x[ok], crew[ok], style, color="tab:red", label=label)
    ax[0].set_xticks(x, bins["horizon"].to_list())
    ax[0].set_xlabel("time to arrival")
    ax[0].set_ylabel("MAE [h]")
    ax[0].set_title("ETA error by prediction horizon (held-out later arrivals)")
    ax[0].legend()
    ax[0].grid(axis="y", alpha=0.3)

    t = test["target_h"].to_numpy()
    lim = float(np.nanmax(t)) * 1.05
    ax[1].scatter(t, test["baseline_h"].clip(upper_bound=lim * 1.5), s=4, alpha=0.3,
                  color="#999", label="baseline")
    ax[1].scatter(t, test["model_h"], s=4, alpha=0.4, color="tab:blue", label="model")
    ax[1].plot([0, lim], [0, lim], "k-", lw=1)
    ax[1].set_xlim(0, lim)
    ax[1].set_ylim(0, lim * 1.5)
    ax[1].set_xlabel("true time to arrival [h]")
    ax[1].set_ylabel("predicted [h]")
    ax[1].set_title("Predicted vs true (test set)")
    ax[1].legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ================================================================ Part B

def scenario_plot(truth, gnss, out_png: Path) -> Path:
    """Simulated passage: true track, GNSS fixes (every 30th), heading and speed."""
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].plot(truth.p[:, 1] / 1000, truth.p[:, 0] / 1000, "k-", lw=1.2, label="truth")
    ax[0].plot(gnss.p[::30, 1] / 1000, gnss.p[::30, 0] / 1000, ".", color="tab:blue", ms=3,
               label="GNSS fix (every 30 s)")
    ax[0].plot(0, 0, "go", label="start")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("East [km]")
    ax[0].set_ylabel("North [km]")
    ax[0].legend()
    ax[0].set_title("Simulated 1 h coastal passage")
    t_min = truth.t / 60
    ax[1].plot(t_min, np.degrees(truth.psi) % 360, color="tab:red", label="heading [deg]")
    ax2 = ax[1].twinx()
    ax2.plot(t_min, truth.u / (1852 / 3600), color="tab:blue", label="speed through water [kn]")
    ax[1].set_xlabel("time [min]")
    ax[1].set_ylabel("heading [deg]", color="tab:red")
    ax2.set_ylabel("speed [kn]", color="tab:blue")
    ax2.set_ylim(0, 15)
    ax[1].set_title("Heading and speed")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def drift_plot(t: np.ndarray, drift: dict, gnss_spec, out_png: Path) -> Path:
    """Unaided INS horizontal error vs time: median and 5-95 % band per IMU grade."""
    plt = _plt()
    colors = {"tactical": "tab:red", "navigation": "tab:blue"}
    fig, ax = plt.subplots(figsize=(10, 6))
    t_min = t / 60
    for grade, err in drift.items():
        c = colors.get(grade, "k")
        ax.fill_between(t_min[1:], np.percentile(err[:, 1:], 5, axis=0),
                        np.percentile(err[:, 1:], 95, axis=0), color=c, alpha=0.15)
        ax.plot(t_min[1:], np.median(err[:, 1:], axis=0), color=c, lw=2,
                label=f"{grade} grade: median (band 5-95 %, {err.shape[0]} runs)")
    ax.axhline(np.sqrt(2) * gnss_spec.sigma_total_m, color="k", ls="--",
               label="GNSS horizontal 1-sigma")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.1)
    ax.set_xlabel("time since GNSS lost [min]")
    ax.set_ylabel("horizontal position error [m]")
    ax.set_title("Free-inertial drift (INS alone, perfect initialisation)")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def allan_plot(tg, ag, ta, aa, spec, out_png: Path) -> Path:
    """Allan deviation of the simulated static gyro and accelerometer vs the datasheet terms."""
    from maritime.nav.sensors import allan_theory

    plt = _plt()
    dph = np.pi / 180 / 3600
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(tg, ag / dph, "k-", lw=1.5, label="simulated gyro")
    ax[0].loglog(tg, spec.gyro_n / np.sqrt(tg) / dph, "r--",
                 label=f"ARW {spec.gyro_arw_dprh} deg/sqrt(h): slope -1/2")
    ax[0].loglog(tg, allan_theory(tg, spec.gyro_n, spec.gyro_bi, spec.gyro_bias_tau_s) / dph,
                 "b:", lw=2, label=f"theory: ARW + Gauss-Markov "
                 f"({spec.gyro_bias_instability_dph} deg/h, tau {spec.gyro_bias_tau_s:.0f} s)")
    ax[0].set_xlabel("averaging time tau [s]")
    ax[0].set_ylabel("Allan deviation [deg/h]")
    ax[0].set_title("Gyro")
    ug = 1e-6 * 9.80665
    ax[1].loglog(ta, aa / ug, "k-", lw=1.5, label="simulated accelerometer")
    ax[1].loglog(ta, spec.accel_n / np.sqrt(ta) / ug, "r--",
                 label=f"VRW {spec.accel_vrw_mpsprh} m/s/sqrt(h)")
    ax[1].loglog(ta, allan_theory(ta, spec.accel_n, spec.accel_bi, spec.accel_bias_tau_s) / ug,
                 "b:", lw=2, label=f"theory: VRW + Gauss-Markov "
                 f"({spec.accel_bias_instability_ug} ug, tau {spec.accel_bias_tau_s:.0f} s)")
    ax[1].set_xlabel("averaging time tau [s]")
    ax[1].set_ylabel("Allan deviation [ug]")
    ax[1].set_title("Accelerometer")
    for a in ax:
        a.grid(which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Error-model check: Allan deviation of a 6 h static record (tactical grade)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def _shade_manoeuvres(ax, truth, t_min) -> None:
    """Grey bands where the ship turns or changes speed (heading becomes observable)."""
    k = (np.abs(truth.w) > 1e-5) | (np.abs(truth.f_b[:, 0]) > 1e-5)
    edges = np.flatnonzero(np.diff(k.astype(int)))
    starts = list(edges[::2] + 1) if k[0] == 0 else [0, *list(edges[1::2] + 1)]
    ends = list(edges[1::2] + 1) if k[0] == 0 else list(edges[::2] + 1)
    for a, b in zip(starts, ends + [len(k) - 1] * (len(starts) - len(ends)), strict=False):
        ax.axvspan(truth.t[a] / 60, truth.t[b] / 60, color="0.85", zorder=0)


def ekf_single_run_plot(truth, mc: dict, out_png: Path) -> Path:
    """One Monte Carlo run: errors against the filter's own +-3 sigma, and bias estimates."""
    plt = _plt()
    log, imu, err, sig = mc["log"], mc["imu"], mc["err"][0], mc["sig"][0]
    t = log.t / 60
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for i, (name, c) in enumerate((("north", "tab:blue"), ("east", "tab:orange"))):
        ax[0, 0].plot(t, err[:, i], color=c, lw=0.8, label=f"{name} error")
        ax[0, 0].plot(t, 3 * sig[:, i], color=c, lw=0.6, ls="--")
        ax[0, 0].plot(t, -3 * sig[:, i], color=c, lw=0.6, ls="--")
    ax[0, 0].set_ylabel("position error [m]")
    ax[0, 0].set_title("Position error and filter +-3 sigma (dashed)")
    ax[0, 0].legend(fontsize=8)

    _shade_manoeuvres(ax[0, 1], truth, t)
    ax[0, 1].plot(t, np.degrees(err[:, 4]), "k", lw=0.8, label="heading error")
    ax[0, 1].plot(t, 3 * np.degrees(sig[:, 4]), "r--", lw=0.8, label="+-3 sigma")
    ax[0, 1].plot(t, -3 * np.degrees(sig[:, 4]), "r--", lw=0.8)
    ax[0, 1].set_ylabel("heading error [deg]")
    ax[0, 1].set_title("Heading: observable only in turns / speed changes (grey)")
    ax[0, 1].legend(fontsize=8)

    mg = 9.80665e-3
    k = log.idx
    for i, c in enumerate(("tab:blue", "tab:orange")):
        ax[1, 0].plot(t, imu.accel_bias[k, i] / mg, color=c, lw=1.5, alpha=0.5,
                      label=f"true accel bias {'xy'[i]}")
        ax[1, 0].plot(t, log.b_a[:, i] / mg, color=c, lw=1, ls="--",
                      label=f"estimated {'xy'[i]}")
    ax[1, 0].set_ylabel("accelerometer bias [mg]")
    ax[1, 0].set_xlabel("time [min]")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("Accelerometer bias estimation")

    dph = np.pi / 180 / 3600
    _shade_manoeuvres(ax[1, 1], truth, t)
    ax[1, 1].plot(t, imu.gyro_bias[k] / dph, "k", lw=1.5, alpha=0.5, label="true gyro bias")
    ax[1, 1].plot(t, log.b_g / dph, "r--", lw=1, label="estimated")
    ax[1, 1].set_ylabel("gyro bias [deg/h]")
    ax[1, 1].set_xlabel("time [min]")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("Gyro bias estimation (learned mainly during manoeuvres)")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle("Loosely coupled INS/GNSS EKF, tactical MEMS IMU, realistic GNSS (one run)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def ekf_consistency_plot(truth, mcs: dict, settle: int, out_png: Path) -> Path:
    """Monte Carlo average NEES and NIS per experiment against the 95 % chi-square band."""
    from scipy.stats import chi2

    plt = _plt()
    colors = ["tab:green", "tab:red", "tab:blue"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for (name, mc), c in zip(mcs.items(), colors, strict=False):
        t = mc["log"].t / 60
        ax[0].plot(t, mc["nees"].mean(axis=0), color=c, lw=1, label=name)
        ax[1].plot(t, mc["nis"].mean(axis=0), color=c, lw=1, label=name)
    runs = next(iter(mcs.values()))["nees"].shape[0]
    for a, dim, label in ((ax[0], 8, "NEES (8 states)"), (ax[1], 2, "NIS (2-D GNSS)")):
        lo, hi = chi2.ppf([0.025, 0.975], runs * dim) / runs
        a.axhspan(lo, hi, color="0.85", label=f"95 % band, {runs} runs")
        a.axhline(dim, color="k", lw=0.8, ls=":")
        a.axvline(settle / 60, color="k", lw=0.5)
        a.set_yscale("log")
        a.set_xlabel("time [min]")
        a.set_ylabel(f"Monte Carlo average {label.split()[0]}")
        a.set_title(f"Average {label}")
        a.grid(which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Filter consistency: white GNSS noise vs realistic (time-correlated) GNSS")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def attack_overview_plot(mcs: dict, attacks: dict, gate: float, out_png: Path,
                         title: str = "Unprotected EKF under GNSS attack") -> Path:
    """One row per attack: horizontal error (median and 5-95 % over runs) and NIS (run 0)."""
    plt = _plt()
    names = list(attacks)
    fig, ax = plt.subplots(len(names), 2, figsize=(14, 3.4 * len(names)), sharex=True)
    for i, name in enumerate(names):
        mc, att = mcs[name], attacks[name]
        t = mc["t"] / 60
        for a in ax[i]:
            a.axvspan(att.start_s / 60, att.end_s / 60, color="tab:red", alpha=0.08)
        lo, med, hi = np.percentile(mc["err"], [5, 50, 95], axis=0)
        ax[i, 0].fill_between(t, lo, hi, color="tab:blue", alpha=0.2)
        ax[i, 0].plot(t, med, color="tab:blue", lw=1.3, label="filter error (median, 5-95 %)")
        if "none" in mcs:
            ax[i, 0].plot(t, np.median(mcs["none"]["err"], axis=0), color="0.4", lw=0.8,
                          ls="--", label="no attack (median)")
        ax[i, 0].set_yscale("log")
        ax[i, 0].set_ylim(0.3, None)
        ax[i, 0].set_ylabel("horizontal error [m]")
        ax[i, 0].set_title(f"{name}: position error")
        ax[i, 0].legend(fontsize=8, loc="upper left")
        nis = mc["nis"][0]
        acc = mc["accepted"][0]
        ax[i, 1].plot(t, nis, color="k", lw=0.6, label="NIS (run 0)")
        if (~acc & ~np.isnan(nis)).any():
            ax[i, 1].plot(t[~acc & ~np.isnan(nis)], nis[~acc & ~np.isnan(nis)], "rx", ms=3,
                          label="fix rejected")
        ax[i, 1].axhline(gate, color="tab:red", ls="--", label=f"chi-square 99 % = {gate}")
        ax[i, 1].set_yscale("log")
        ax[i, 1].set_ylabel("NIS")
        ax[i, 1].set_title(f"{name}: innovation test statistic")
        ax[i, 1].legend(fontsize=8, loc="upper left")
    for a in ax[-1]:
        a.set_xlabel("time [min]  (red band = attack window)")
    for a in ax.flat:
        a.grid(which="both", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def drift_stealth_plot(sweep: pl.DataFrame, false_alarm_pct: float, out_png: Path) -> Path:
    """How visible is a slow drift to the per-epoch chi-square test, vs how far it pulls."""
    plt = _plt()
    r = sweep["rate_mps"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(r, sweep["nis_over_gate_pct"], "o-", color="tab:red",
            label="attack epochs with NIS > chi-square 99 %")
    ax.axhline(false_alarm_pct, color="0.4", ls="--",
               label=f"same rate with no attack ({false_alarm_pct:.1f} %)")
    ax.set_xscale("log")
    ax.set_xlabel("spoofed drift rate [m/s]")
    ax.set_ylabel("per-epoch detections [%]", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(r, sweep["filter_err_at_end_m"], "s--", color="tab:blue",
             label="filter error at end of 15 min attack")
    ax2.set_ylabel("filter position error at end [m]", color="tab:blue")
    ax.set_title("Slow GNSS drift: the filter follows it, the per-epoch test barely notices")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def protection_comparison_plot(mcs_u: dict, mcs_p: dict, attacks: dict, out_png: Path) -> Path:
    """Per attack: median horizontal error with and without integrity monitoring."""
    plt = _plt()
    names = list(attacks)
    fig, ax = plt.subplots(len(names), 1, figsize=(12, 3.3 * len(names)), sharex=True)
    for a, name in zip(ax, names, strict=True):
        att, mu, mp = attacks[name], mcs_u[name], mcs_p[name]
        t = mu["t"] / 60
        a.axvspan(att.start_s / 60, att.end_s / 60, color="tab:red", alpha=0.08)
        for mc, color, label in ((mu, "tab:red", "unprotected"), (mp, "tab:blue", "protected")):
            lo, med, hi = np.percentile(mc["err"], [5, 50, 95], axis=0)
            a.fill_between(t, lo, hi, color=color, alpha=0.15)
            a.plot(t, med, color=color, lw=1.4, label=f"{label} (median, 5-95 %)")
        dets = [next((e.t for e in ev if e.kind == "declared"), None) for ev in mp["events"]]
        dets = [d for d in dets if d is not None and d >= att.start_s]
        if dets:
            a.axvline(np.median(dets) / 60, color="k", ls="--", lw=1,
                      label=f"median detection (+{np.median(dets) - att.start_s:.0f} s)")
        a.set_yscale("log")
        a.set_ylim(0.3, None)
        a.set_ylabel("horizontal error [m]")
        a.set_title(name)
        a.grid(which="both", alpha=0.3)
        a.legend(fontsize=8, loc="upper left")
    ax[-1].set_xlabel("time [min]  (red band = attack window)")
    fig.suptitle("Integrity monitoring: chi-square gate + backup-INS window test, "
                 "rollback and speed-log coasting")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def window_calibration_plot(cal: dict, out_png: Path) -> Path:
    """Attack-free distribution of the backup-INS test statistic vs theory and threshold."""
    from scipy.stats import chi2

    plt = _plt()
    stats, alpha, thr = cal["stats"], cal["alpha"], cal["threshold"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(0, max(stats.max(), thr) * 1.05, 60)
    ax.hist(stats, bins=bins, density=True, color="0.6",
            label=f"attack-free window tests (n = {stats.size})")
    x = np.linspace(0.01, bins[-1], 400)
    ax.plot(x, chi2.pdf(x, 2), "k--", label="chi-square(2): white-noise theory (mean 2)")
    ax.plot(x, chi2.pdf(x / alpha, 2) / alpha, "tab:blue", lw=2,
            label=f"fitted alpha x chi-square(2), alpha = {alpha:.2f}")
    ax.axvline(thr, color="tab:red", lw=2, label=f"threshold {thr:.1f} (pfa 1e-4 per test)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("T = r' S^-1 r  (GNSS vs 2-min backup INS)")
    ax.set_ylabel("density (log)")
    ax.set_title("Calibrating the drift detector on attack-free Monte Carlo")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def detection_curve_plot(rows: list[dict], out_png: Path) -> Path:
    """Detection probability (95 % Wilson CI) and median delay vs attack size, per IMU grade."""
    plt = _plt()
    colors = {"tactical": "tab:red", "navigation": "tab:blue"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5), sharex="col")
    for col, kind, xlabel in ((0, "drift", "spoofed drift rate [m/s]"),
                              (1, "jump", "spoofed jump size [m]")):
        for grade in sorted({r["grade"] for r in rows if r["attack"] == kind}):
            pts = sorted((r for r in rows if r["attack"] == kind and r["grade"] == grade),
                         key=lambda r: r["level"])
            x = np.array([r["level"] for r in pts])
            p = np.array([r["p_detect"] for r in pts])
            lo = p - np.array([r["p_detect_ci95"][0] for r in pts])
            hi = np.array([r["p_detect_ci95"][1] for r in pts]) - p
            c = colors.get(grade, "k")
            ax[0, col].errorbar(x, 100 * p, yerr=100 * np.vstack([lo, hi]), fmt="o-", color=c,
                                capsize=3, label=f"{grade} IMU ({pts[0]['runs']} runs/point)")
            d = [r["delay_median_s"] for r in pts]
            ok = [v is not None for v in d]
            ax[1, col].plot(x[ok], [v for v in d if v is not None], "s-", color=c, label=grade)
        ax[0, col].set_ylabel("detected during attack [%]")
        ax[0, col].set_ylim(-5, 105)
        ax[0, col].set_title(f"{kind}: probability of detection (95 % CI)")
        ax[1, col].set_ylabel("median detection delay [s]")
        ax[1, col].set_xlabel(xlabel)
        ax[1, col].set_title(f"{kind}: time to detect")
        for a in ax[:, col]:
            a.set_xscale("log")
            a.grid(which="both", alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle("Integrity monitor performance (calibrated per IMU grade, p_fa 1e-4 per test)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def error_reduction_plot(rows: list[dict], out_png: Path) -> Path:
    """Median position error at the end of the attack, unprotected vs protected."""
    plt = _plt()
    colors = {"tactical": "tab:red", "navigation": "tab:blue"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for col, kind, xlabel in ((0, "drift", "spoofed drift rate [m/s]"),
                              (1, "jump", "spoofed jump size [m]")):
        for grade in sorted({r["grade"] for r in rows if r["attack"] == kind}):
            pts = sorted((r for r in rows if r["attack"] == kind and r["grade"] == grade),
                         key=lambda r: r["level"])
            x = [r["level"] for r in pts]
            c = colors.get(grade, "k")
            ax[col].plot(x, [r["err_end_unprotected_m"] for r in pts], "--", color=c, alpha=0.6,
                         label=f"{grade}: unprotected")
            ax[col].plot(x, [r["err_end_protected_m"] for r in pts], "o-", color=c,
                         label=f"{grade}: protected")
        ax[col].set_xscale("log")
        ax[col].set_yscale("log")
        ax[col].set_xlabel(xlabel)
        ax[col].set_ylabel("median position error at end of 15-min attack [m]")
        ax[col].set_title(kind)
        ax[col].grid(which="both", alpha=0.3)
        ax[col].legend(fontsize=8)
    fig.suptitle("What the integrity monitor saves: error at the end of the attack")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def gap_category_chart(gaps: pl.DataFrame, out_png: Path) -> Path:
    """Left: why gaps were explained away. Right: duration distribution of dark gaps."""
    plt = _plt()
    order = ["port_or_anchorage", "low_coverage", "position_jump", "dark"]
    counts = gaps["category"].value_counts()
    n = [int(counts.filter(pl.col("category") == c)["count"].sum()) for c in order]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bars = ax[0].barh(order, n, color=["#999", "#bbb", "#e69f00", "#d62728"])
    ax[0].bar_label(bars, padding=3)
    ax[0].set_xlabel("AIS gaps >= threshold")
    ax[0].set_title("Explanation assigned to each gap")
    dark = gaps.filter(pl.col("category") == "dark")["gap_h"].to_numpy()
    if dark.size:
        ax[1].hist(dark, bins=np.logspace(np.log10(max(dark.min(), 1)), np.log10(dark.max()), 25),
                   color="#d62728")
        ax[1].set_xscale("log")
    ax[1].set_xlabel("Gap duration [h] (log)")
    ax[1].set_ylabel("Dark gaps")
    ax[1].set_title("Duration of unexplained (dark) gaps")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def density_map(positions_path: Path, bbox: BBox, out_png: Path, title: str,
                bins_per_deg: int = 40) -> Path:
    """Log-scaled 2D histogram of AIS fixes: a quick sanity check that the data draws the
    shipping lanes and coastline (no land shapefile needed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    lon_min, lat_min, lon_max, lat_max = bbox
    df = pl.read_parquet(positions_path, columns=["lat", "lon"])
    xbins = int((lon_max - lon_min) * bins_per_deg)
    ybins = int((lat_max - lat_min) * bins_per_deg / np.cos(np.radians((lat_min + lat_max) / 2)))

    h, xe, ye = np.histogram2d(df["lon"].to_numpy(), df["lat"].to_numpy(),
                               bins=[xbins, ybins], range=[[lon_min, lon_max], [lat_min, lat_max]])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(xe, ye, np.where(h > 0, h, np.nan).T, norm=LogNorm(), cmap="viridis")
    # 1 deg of longitude is cos(lat) shorter than 1 deg of latitude: fix the aspect
    ax.set_aspect(1 / np.cos(np.radians((lat_min + lat_max) / 2)))
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="AIS fixes per cell (log)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
