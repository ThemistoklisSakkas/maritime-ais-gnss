"""Part B orchestration, one `run_bX` function per phase."""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from maritime.config import Config
from maritime.nav.ins import free_inertial_batch
from maritime.nav.sensors import (
    GNSSSpec,
    IMUSpec,
    allan_deviation,
    simulate_gnss,
    simulate_imu,
)
from maritime.nav.trajectory import Truth, legs_from_config, simulate_truth


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scenario_truth(cfg: Config) -> Truth:
    s = cfg.section("sim")
    return simulate_truth(legs_from_config(s["legs"]), s["u0_kn"], s["psi0_deg"],
                          s["dt_imu_s"], s["substeps"], tuple(s["current_ne_mps"]))


# ---------------------------------------------------------------- B1

def free_inertial_drift(truth: Truth, spec: IMUSpec, runs: int, seed: int) -> np.ndarray:
    """Horizontal position error (runs, n) of unaided INS started from the true state."""
    rng = np.random.default_rng(seed)
    imus = [simulate_imu(truth, spec, rng) for _ in range(runs)]
    f_b = np.stack([m.f_b for m in imus])
    w = np.stack([m.w for m in imus])
    p, _, _ = free_inertial_batch(truth.p[0], truth.v[0], truth.psi[0], f_b, w, truth.dt)
    return np.linalg.norm(p - truth.p[None], axis=2)


def run_b1(cfg: Config) -> None:
    from maritime.plots import allan_plot, drift_plot, scenario_plot

    s, b1 = cfg.section("sim"), cfg.section("b1")
    tables, figs = cfg.paths.tables, cfg.paths.figures
    cfg.paths.ensure()
    truth = scenario_truth(cfg)
    gnss_spec = GNSSSpec.from_dict(s["gnss"])
    gnss = simulate_gnss(truth, gnss_spec, np.random.default_rng(cfg.seed))
    _log(f"truth: {truth.t[-1] / 60:.0f} min, {len(truth.t)} IMU samples, "
         f"track {np.sum(np.linalg.norm(np.diff(truth.p, axis=0), axis=1)) / 1852:.1f} NM")
    scenario_plot(truth, gnss, figs / "b1_scenario.png")

    # ---- free-inertial drift per grade
    rows, drift = [], {}
    for k, (grade, d) in enumerate(s["imu_grades"].items()):
        t0 = time.perf_counter()
        err = free_inertial_drift(truth, IMUSpec.from_dict(d), b1["mc_runs"], cfg.seed + k)
        drift[grade] = err
        for t_rep in b1["report_times_s"]:
            i = int(round(t_rep / truth.dt))
            rows.append({"grade": grade, "t_s": t_rep,
                         "median_m": float(np.median(err[:, i])),
                         "p95_m": float(np.percentile(err[:, i], 95))})
        _log(f"{grade}: {b1['mc_runs']} runs in {time.perf_counter() - t0:.1f} s")
    table = pl.DataFrame(rows)
    table.write_csv(tables / "b1_free_inertial_drift.csv")
    _log(f"free-inertial horizontal error:\n{table}")
    drift_plot(truth.t, drift, gnss_spec, figs / "b1_free_inertial_drift.png")

    # ---- Allan deviation of the simulated static IMU (validates the error model)
    spec = IMUSpec.from_dict(s["imu_grades"][s["imu_grade"]])
    static = simulate_truth(legs_from_config([{"duration_s": b1["allan_hours"] * 3600}]),
                            0.0, 0.0, s["dt_imu_s"], 1)
    imu = simulate_imu(static, spec, np.random.default_rng(cfg.seed + 100))
    tg, ag = allan_deviation(imu.w - imu.w.mean(), s["dt_imu_s"])
    ta, aa = allan_deviation(imu.f_b[:, 0] - imu.f_b[:, 0].mean(), s["dt_imu_s"])
    allan_plot(tg, ag, ta, aa, spec, figs / "b1_allan_deviation.png")
    arw_est = float(np.interp(1.0, tg, ag)) / spec.gyro_n * spec.gyro_arw_dprh
    vrw_est = float(np.interp(1.0, ta, aa)) / spec.accel_n * spec.accel_vrw_mpsprh

    summary = {
        "scenario_minutes": round(truth.t[-1] / 60, 1),
        "gnss_sigma_per_axis_m": round(gnss_spec.sigma_total_m, 2),
        "drift": {f"{r['grade']}@{r['t_s']}s": {"median_m": round(r["median_m"], 1),
                                                "p95_m": round(r["p95_m"], 1)} for r in rows},
        "allan_check": {
            "gyro_arw_spec_dprh": spec.gyro_arw_dprh, "gyro_arw_recovered_dprh": round(arw_est, 4),
            "accel_vrw_spec": spec.accel_vrw_mpsprh, "accel_vrw_recovered": round(vrw_est, 4),
            "gyro_adev_min_dph": round(float(ag.min()) / (np.pi / 180 / 3600), 3),
        },
    }
    (tables / "b1_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"summary: {json.dumps(summary, indent=2)}")
    _log(f"figures in {figs}")
