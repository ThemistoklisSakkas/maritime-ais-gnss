"""Part B orchestration, one `run_bX` function per phase."""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from maritime.config import Config
from maritime.nav.ekf import EKFParams, error_states, nees, run_ekf
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


# ---------------------------------------------------------------- B2

def ekf_params(cfg: Config, imu: IMUSpec, gnss: GNSSSpec, sigma_scale: float | None = None
               ) -> EKFParams:
    d = dict(cfg.section("ekf"))
    if sigma_scale is not None:
        d["gnss_sigma_scale"] = sigma_scale
    return EKFParams.from_specs(imu, gnss, d)


def monte_carlo_ekf(truth: Truth, imu_spec: IMUSpec, gnss_spec: GNSSSpec, params: EKFParams,
                    runs: int, seed: int) -> dict:
    """Run the EKF `runs` times; return per-run errors, NEES, NIS, sigmas and raw GNSS error."""
    out = {"err": [], "nees": [], "nis": [], "sig": [], "gnss_err": [], "log": None}
    for r in range(runs):
        rng = np.random.default_rng(seed + r)
        imu = simulate_imu(truth, imu_spec, rng)
        gnss = simulate_gnss(truth, gnss_spec, rng)
        log = run_ekf(truth, imu, gnss.t, gnss.idx, gnss.p, params, rng)
        err = error_states(log, truth, imu)
        out["err"].append(err)
        out["nees"].append(nees(err, log.P))
        out["nis"].append(log.nis)
        out["sig"].append(np.sqrt(log.P_diag))
        out["gnss_err"].append(gnss.p - truth.p[gnss.idx])
        if r == 0:
            out["log"], out["imu"] = log, imu
    for k in ("err", "nees", "nis", "sig", "gnss_err"):
        out[k] = np.stack(out[k])
    return out


def summarize_mc(mc: dict, settle: int, n_states: int) -> dict:
    from scipy.stats import chi2

    runs = mc["nees"].shape[0]
    e = mc["err"][:, settle:]
    horiz = np.sqrt(np.sum(e[..., 0:2] ** 2, axis=-1))
    raw = np.sqrt(np.sum(mc["gnss_err"][:, settle:] ** 2, axis=-1))
    avg_nees = mc["nees"][:, settle:].mean(axis=0)
    avg_nis = mc["nis"][:, settle:].mean(axis=0)
    lo, hi = chi2.ppf([0.025, 0.975], runs * n_states) / runs
    lo2, hi2 = chi2.ppf([0.025, 0.975], runs * 2) / runs
    return {
        "pos_rms_m": round(float(np.sqrt(np.mean(horiz ** 2))), 2),
        "raw_gnss_rms_m": round(float(np.sqrt(np.mean(raw ** 2))), 2),
        "heading_rms_deg": round(float(np.degrees(np.sqrt(np.mean(e[..., 4] ** 2)))), 3),
        "accel_bias_rms_mg": round(float(np.sqrt(np.mean(e[..., 5:7] ** 2)) / 9.80665e-3), 4),
        "gyro_bias_rms_dph": round(float(np.sqrt(np.mean(e[..., 7] ** 2))
                                         / (np.pi / 180 / 3600)), 3),
        "avg_nees": round(float(avg_nees.mean()), 2),
        "nees_epochs_above_band_pct": round(100 * float((avg_nees > hi).mean()), 1),
        "nees_epochs_in_band_pct": round(100 * float(((avg_nees > lo) & (avg_nees < hi)).mean()),
                                         1),
        "avg_nis": round(float(avg_nis.mean()), 2),
        "nis_epochs_in_band_pct": round(100 * float(((avg_nis > lo2) & (avg_nis < hi2)).mean()),
                                        1),
    }


def run_b2(cfg: Config) -> None:
    from maritime.plots import ekf_consistency_plot, ekf_single_run_plot

    s, b2 = cfg.section("sim"), cfg.section("b2")
    tables, figs = cfg.paths.tables, cfg.paths.figures
    cfg.paths.ensure()
    truth = scenario_truth(cfg)
    imu_spec = IMUSpec.from_dict(s["imu_grades"][s["imu_grade"]])
    base_gnss = s["gnss"]
    settle = int(b2["settle_s"])

    results, mcs = {}, {}
    for ex in b2["experiments"]:
        g = dict(base_gnss)
        if "gnss_corr_m" in ex:          # same total noise, but all of it white
            total = GNSSSpec.from_dict(base_gnss).sigma_total_m
            g["sigma_corr_m"], g["sigma_white_m"] = ex["gnss_corr_m"], total
        gnss_spec = GNSSSpec.from_dict(g)
        params = ekf_params(cfg, imu_spec, gnss_spec, ex["gnss_sigma_scale"])
        t0 = time.perf_counter()
        mc = monte_carlo_ekf(truth, imu_spec, gnss_spec, params, b2["mc_runs"], cfg.seed)
        mcs[ex["name"]] = mc
        results[ex["name"]] = summarize_mc(mc, settle, 8)
        _log(f"{ex['name']}: {b2['mc_runs']} runs in {time.perf_counter() - t0:.0f} s -> "
             f"{results[ex['name']]}")

    summary = {"imu_grade": s["imu_grade"], "runs_per_experiment": b2["mc_runs"],
               "settle_s": settle, "experiments": results}
    (tables / "b2_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"summary: {json.dumps(summary, indent=2)}")

    ekf_single_run_plot(truth, mcs["realistic_gnss"], figs / "b2_single_run.png")
    ekf_consistency_plot(truth, mcs, settle, figs / "b2_consistency.png")
    _log(f"figures in {figs}")
