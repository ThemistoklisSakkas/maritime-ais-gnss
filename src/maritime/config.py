"""Load config/default.yaml into typed, immutable objects.

Paths in the YAML are relative to the repository root and are resolved here,
so the pipeline behaves the same regardless of the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"

BBox = tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max


@dataclass(frozen=True)
class Paths:
    raw: Path
    reference: Path
    interim: Path
    processed: Path
    figures: Path
    tables: Path

    def ensure(self) -> None:
        """Create all output directories if missing."""
        for p in (self.raw, self.reference, self.interim, self.processed,
                  self.figures, self.tables):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    start: date
    end: date
    aoi: dict[str, BBox]
    crs_metric: str
    raw: dict[str, Any] = field(repr=False)  # full YAML, for phase-specific sections

    @property
    def days(self) -> list[date]:
        """All study days, inclusive of start and end."""
        n = (self.end - self.start).days + 1
        return [self.start + timedelta(days=i) for i in range(n)]

    def section(self, name: str) -> dict[str, Any]:
        """Return a phase-specific section (e.g. 'cleaning'); fails loudly if missing."""
        if name not in self.raw:
            raise KeyError(f"config section '{name}' not found")
        return self.raw[name]


def _parse_bbox(values: list[float]) -> BBox:
    if len(values) != 4:
        raise ValueError(f"bbox must have 4 values, got {values}")
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in values)
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError(f"bbox min must be < max: {values}")
    return lon_min, lat_min, lon_max, lat_max


def load_config(path: str | Path | None = None, root: Path = REPO_ROOT) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths = Paths(**{k: (root / v).resolve() for k, v in raw["paths"].items()})
    start = date.fromisoformat(str(raw["dates"]["start"]))
    end = date.fromisoformat(str(raw["dates"]["end"]))
    if end < start:
        raise ValueError("dates.end must not be before dates.start")

    return Config(
        seed=int(raw["seed"]),
        paths=paths,
        start=start,
        end=end,
        aoi={name: _parse_bbox(v) for name, v in raw["aoi"].items()},
        crs_metric=str(raw["crs_metric"]),
        raw=raw,
    )
