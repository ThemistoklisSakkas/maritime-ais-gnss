from datetime import date

import pytest
import yaml

from maritime.config import load_config

MINIMAL = {
    "seed": 7,
    "paths": {k: f"data/{k}" for k in
              ("raw", "reference", "interim", "processed", "figures", "tables")},
    "dates": {"start": "2025-01-30", "end": "2025-02-02"},
    "aoi": {"full": [3.0, 53.5, 17.0, 58.5]},
    "crs_metric": "EPSG:25832",
}


def write(tmp_path, cfg):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_default_config_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.paths.raw.is_absolute()
    assert len(cfg.days) == 5


def test_days_inclusive_across_month(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL), root=tmp_path)
    assert cfg.days == [date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2)]
    assert cfg.paths.raw == (tmp_path / "data" / "raw").resolve()


def test_invalid_bbox_rejected(tmp_path):
    bad = {**MINIMAL, "aoi": {"full": [17.0, 53.5, 3.0, 58.5]}}
    with pytest.raises(ValueError):
        load_config(write(tmp_path, bad), root=tmp_path)


def test_end_before_start_rejected(tmp_path):
    bad = {**MINIMAL, "dates": {"start": "2025-02-02", "end": "2025-01-30"}}
    with pytest.raises(ValueError):
        load_config(write(tmp_path, bad), root=tmp_path)


def test_missing_section_fails_loudly():
    with pytest.raises(KeyError):
        load_config().section("does_not_exist")
