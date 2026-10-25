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
