"""Static figures for the README (matplotlib, saved as PNG)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from maritime.config import BBox


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
