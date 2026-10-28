import math

import numpy as np
import pytest

from maritime.config import load_config
from maritime.nav.attacks import Attack
from maritime.nav.montecarlo import Task, run_parallel, run_task, summarize_point, wilson_interval


def short_cfg(tmp_path):
    """Default config with a 20-min scenario so the worker tests stay fast."""
    import yaml

    from maritime.config import DEFAULT_CONFIG

    d = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    d["sim"]["legs"] = [{"duration_s": 600}, {"duration_s": 180, "turn_rate_dps": 0.5},
                        {"duration_s": 420}]
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return load_config(p)


# ---------------------------------------------------------------- statistics

@pytest.mark.parametrize("k, n", [(0, 10), (5, 10), (10, 10), (37, 100)])
def test_wilson_interval_contains_estimate_and_stays_in_unit_range(k, n):
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)


def test_wilson_interval_empty():
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_summarize_point():
    pts = [{"detected": True, "delay_s": 100.0, "false_before": False,
            "err_end_u": 180.0, "err_end_p": 30.0},
           {"detected": False, "delay_s": None, "false_before": False,
            "err_end_u": 180.0, "err_end_p": 180.0}]
    s = summarize_point(pts)
    assert s["detected"] == 1 and s["p_detect"] == 0.5
    assert s["delay_median_s"] == 100.0
    assert s["err_end_protected_m"] == pytest.approx(105.0)


# ---------------------------------------------------------------- workers

def test_paired_runs_share_sensors_and_initial_error(tmp_path):
    """With no attack and no alarm, the protected and unprotected filters must coincide:
    proof that both see identical IMU, GNSS and initial errors."""
    cfg = short_cfg(tmp_path)
    r = run_task(cfg, Task("tactical", 11, None, threshold=1e9))
    assert r["declarations"] == []
    assert r["mean_err_p"] == pytest.approx(r["mean_err_u"], rel=1e-9)


def test_jump_task_reports_detection(tmp_path):
    cfg = short_cfg(tmp_path)
    att = Attack("jump", 900, 200, 90, magnitude_m=180)
    r = run_task(cfg, Task("tactical", 12, att, threshold=64.0))
    assert r["detected"] and r["delay_s"] <= 5
    assert r["err_end_p"] < r["err_end_u"]


def test_parallel_equals_serial(tmp_path):
    cfg = short_cfg(tmp_path)
    tasks = [Task("tactical", s, None, threshold=1e9, protected_only=True) for s in (1, 2)]
    serial = run_parallel(cfg, tasks, workers=1)
    parallel = run_parallel(cfg, tasks, workers=2)
    assert [r["mean_err_p"] for r in serial] == pytest.approx(
        [r["mean_err_p"] for r in parallel], rel=1e-12)
    assert np.isfinite([r["mean_err_p"] for r in serial]).all()
