"""Download and extract Danish Maritime Authority (DMA) daily AIS files.

The DMA bucket stores one zip per day (~650 MB, ~3.3 GB CSV inside). Older
days live under a year prefix ("2025/aisdk-2025-01-01.zip"), recent days at the
bucket root ("aisdk-2025-06-02.zip"), so both locations are tried.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import date
from pathlib import Path

import requests
from tqdm import tqdm

CHUNK = 1 << 20  # 1 MiB


def day_filename(day: date) -> str:
    return f"aisdk-{day.isoformat()}.zip"


def day_url_candidates(day: date, base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    name = day_filename(day)
    return [f"{base}/{name}", f"{base}/{day.year}/{name}"]


def _remote_size(url: str, timeout: float = 30) -> int | None:
    r = requests.head(url, timeout=timeout, allow_redirects=True)
    if r.status_code != 200:
        return None
    return int(r.headers.get("Content-Length", 0)) or None


def download_day(day: date, raw_dir: Path, base_url: str) -> Path:
    """Download one day's zip into raw_dir. Skips if a complete copy already exists."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / day_filename(day)

    for url in day_url_candidates(day, base_url):
        size = _remote_size(url)
        if size is None:
            continue
        if dest.exists() and dest.stat().st_size == size:
            return dest  # already complete

        tmp = dest.with_suffix(".part")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f, tqdm(total=size, unit="B", unit_scale=True,
                                            desc=dest.name) as bar:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
                    bar.update(len(chunk))
        if tmp.stat().st_size != size:
            raise OSError(f"incomplete download for {url}")
        tmp.replace(dest)  # atomic: a crash never leaves a "complete-looking" file
        return dest

    raise FileNotFoundError(f"no DMA file found for {day} under {base_url}")


def extract_csv(zip_path: Path, out_dir: Path) -> Path:
    """Stream the single CSV out of the zip (constant memory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one CSV in {zip_path}, found {members}")
        dest = out_dir / Path(members[0]).name
        with zf.open(members[0]) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
    return dest
