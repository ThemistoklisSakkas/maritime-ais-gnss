"""B5 - Parallel Monte Carlo evaluation of the integrity monitor.

Every run is independent (own seed), so the study is "embarrassingly parallel": runs are
farmed out to a process pool. Each worker rebuilds the scenario from the (picklable) config
and returns a small dict of numbers, never large arrays.

Pairing: within a run, the IMU and GNSS come from `default_rng(seed)`, the filter's initial
error from `default_rng([seed, 2])` and the speed log from `default_rng([seed, 1])`. The
unprotected and protected filters therefore see exactly the same sensors and start from the
same initial error, so their difference is due to the monitor alone.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from maritime.config import Config
from maritime.nav.attacks import Attack, apply_attack
from maritime.nav.ekf import EKFParams, run_ekf
from maritime.nav.integrity import IntegrityMonitor, MonitorParams
from maritime.nav.sensors import (
    GNSSSpec,
    IMUSpec,
    SpeedLogSpec,
    simulate_gnss,
    simulate_imu,
    simulate_speed_log,
)
from maritime.nav.trajectory import Truth, legs_from_config, simulate_truth


@dataclass(frozen=True)
class Task:
    grade: str
    seed: int
    attack: Attack | None
    threshold: float            # window-test threshold for this grade (inf when calibrating)
    calibrate: bool = False
    protected_only: bool = False


_TRUTH_CACHE: dict[str, Truth] = {}


def _truth(cfg: Config) -> Truth:
    """Scenario truth, built once per worker process (keyed by the scenario parameters)."""
    s = cfg.section("sim")
    key = json.dumps({k: s[k] for k in ("legs", "u0_kn", "psi0_deg", "dt_imu_s", "substeps",
                                        "current_ne_mps")}, sort_keys=True)
    if key not in _TRUTH_CACHE:
        _TRUTH_CACHE[key] = simulate_truth(legs_from_config(s["legs"]), s["u0_kn"],
                                           s["psi0_deg"], s["dt_imu_s"], s["substeps"],
                                           tuple(s["current_ne_mps"]))
    return _TRUTH_CACHE[key]


def _specs(cfg: Config, grade: str) -> tuple[IMUSpec, GNSSSpec, EKFParams]:
    s = cfg.section("sim")
    imu = IMUSpec.from_dict(s["imu_grades"][grade])
    gnss = GNSSSpec.from_dict(s["gnss"])
    return imu, gnss, EKFParams.from_specs(imu, gnss, cfg.section("ekf"))


def run_task(cfg: Config, task: Task) -> dict:
    """One Monte Carlo run: unprotected and protected filters on identical data."""
    truth = _truth(cfg)
    imu_spec, gnss_spec, params = _specs(cfg, task.grade)
    rng = np.random.default_rng(task.seed)
    imu = simulate_imu(truth, imu_spec, rng)
    gnss, mask = apply_attack(simulate_gnss(truth, gnss_spec, rng), task.attack)
    log_u = simulate_speed_log(truth, SpeedLogSpec.from_dict(cfg.section("sim")["speed_log"]),
                               np.random.default_rng([task.seed, 1])).u

    mon = IntegrityMonitor(MonitorParams.from_dict(cfg.section("integrity"), task.threshold),
                           imu, gnss.t, gnss.idx, truth.dt, speed_log=log_u,
                           calibrate=task.calibrate)
    log_p = run_ekf(truth, imu, gnss.t, gnss.idx, gnss.p, params,
                    np.random.default_rng([task.seed, 2]), monitor=mon)
    out = {"grade": task.grade, "seed": task.seed,
           "window_stats": [T for t, T in mon.window_stats] if task.calibrate else None,
           "window_stats_t": [t for t, T in mon.window_stats] if task.calibrate else None,
           "declarations": [(e.t, e.cause) for e in mon.events if e.kind == "declared"],
           "err_p": np.linalg.norm(log_p.p - truth.p[log_p.idx], axis=1)}
    if not task.protected_only:
        log_u_ = run_ekf(truth, imu, gnss.t, gnss.idx, gnss.p, params,
                         np.random.default_rng([task.seed, 2]))
        out["err_u"] = np.linalg.norm(log_u_.p - truth.p[log_u_.idx], axis=1)
    out["t"] = log_p.t
    return _compact(out, task, mask)


def _compact(out: dict, task: Task, mask: np.ndarray) -> dict:
    """Reduce the error series to the few numbers the study needs (keeps IPC small)."""
    t, att = out.pop("t"), task.attack
    err_p, err_u = out.pop("err_p"), out.pop("err_u", None)
    if att is None:
        sel = t >= 600
        out["mean_err_p"] = float(err_p[sel].mean())
        if err_u is not None:
            out["mean_err_u"] = float(err_u[sel].mean())
        return out
    end_i = int(np.flatnonzero(mask)[-1])
    after = (t >= att.end_s) & (t < att.end_s + 300)
    first = next(((ti, c) for ti, c in out["declarations"]
                  if att.start_s <= ti < att.end_s), None)
    out.update({
        "detected": first is not None,
        "delay_s": None if first is None else first[0] - att.start_s,
        "cause": None if first is None else first[1],
        "false_before": any(ti < att.start_s for ti, _ in out["declarations"]),
        "err_end_p": float(err_p[end_i]), "err_max_p": float(err_p[mask].max()),
        "err_after_p": float(err_p[after].mean()),
    })
    if err_u is not None:
        out.update({"err_end_u": float(err_u[end_i]), "err_max_u": float(err_u[mask].max()),
                    "err_after_u": float(err_u[after].mean())})
    return out


def run_parallel(cfg: Config, tasks: list[Task], workers: int) -> list[dict]:
    """Map tasks over a process pool (or serially with workers <= 1, e.g. in tests)."""
    if workers <= 1:
        return [run_task(cfg, t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_task, [cfg] * len(tasks), tasks, chunksize=1))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a detection probability k/n (sane at 0/n and n/n)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_point(results: list[dict]) -> dict:
    """Detection probability (with CI), delay and paired errors for one sweep point."""
    n = len(results)
    k = sum(r["detected"] for r in results)
    lo, hi = wilson_interval(k, n)
    delays = [r["delay_s"] for r in results if r["detected"]]

    def med(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(float(np.median(vals)), 1) if vals else None
    return {
        "runs": n, "detected": k, "p_detect": round(k / n, 3),
        "p_detect_ci95": [round(lo, 3), round(hi, 3)],
        "delay_median_s": round(float(np.median(delays)), 1) if delays else None,
        "delay_p90_s": round(float(np.percentile(delays, 90)), 1) if delays else None,
        "false_alarm_before_attack_runs": sum(r["false_before"] for r in results),
        "err_end_unprotected_m": med("err_end_u"), "err_end_protected_m": med("err_end_p"),
        "err_max_unprotected_m": med("err_max_u"), "err_max_protected_m": med("err_max_p"),
        "err_after_unprotected_m": med("err_after_u"),
        "err_after_protected_m": med("err_after_p"),
    }


# ---------------------------------------------------------------- B5 study

def run_b5(cfg: Config) -> None:
    import time

    import polars as pl

    from maritime.nav.integrity import calibrate_window_threshold
    from maritime.plots import detection_curve_plot, error_reduction_plot

    b5, ic = cfg.section("b5"), cfg.section("integrity")
    a = cfg.section("attacks")
    tables, figs = cfg.paths.tables, cfg.paths.figures
    cfg.paths.ensure()
    workers, runs = int(b5["workers"]), int(b5["runs_per_point"])
    grades = list(b5["grades"])

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # 1) calibrate the window test separately for each IMU grade (independent seeds)
    t0 = time.perf_counter()
    cal_tasks = [Task(g, cfg.seed + 5000 + r, None, math.inf, calibrate=True,
                      protected_only=True) for g in grades for r in range(int(b5["calib_runs"]))]
    cal_res = run_parallel(cfg, cal_tasks, workers)
    thresholds, calibration = {}, {}
    for g in grades:
        stats = np.array([T for r in cal_res if r["grade"] == g
                          for T, t in zip(r["window_stats"], r["window_stats_t"], strict=True)
                          if t >= 600])
        thr, alpha = calibrate_window_threshold(stats, float(ic["pfa_window"]))
        thresholds[g] = thr
        calibration[g] = {"threshold": round(thr, 1), "alpha": round(alpha, 2),
                          "n_tests": int(stats.size),
                          "exceed_pct": round(100 * float(np.mean(stats > thr)), 3)}
    log(f"calibration in {time.perf_counter() - t0:.0f} s: {calibration}")

    # 2) the study: drift sweep (all grades), jump sweep, attack-free runs for false alarms
    start, dur, direc = float(a["start_s"]), float(a["duration_s"]), float(a["direction_deg"])
    base = cfg.seed + int(b5["seed_offset"])
    tasks, keys = [], []
    for g in grades:
        for rate in b5["drift_rates_mps"]:
            for r in range(runs):            # same seeds at every rate: common random numbers
                tasks.append(Task(g, base + r, Attack("drift", start, dur, direc,
                                                      rate_mps=float(rate)), thresholds[g]))
                keys.append(("drift", g, float(rate)))
    for g in b5["jump_grades"]:
        for mag in b5["jump_magnitudes_m"]:
            for r in range(runs):
                tasks.append(Task(g, base + r, Attack("jump", start, dur, direc,
                                                      magnitude_m=float(mag)), thresholds[g]))
                keys.append(("jump", g, float(mag)))
    for g in grades:
        for r in range(int(b5["false_alarm_runs"])):
            tasks.append(Task(g, cfg.seed + 9000 + r, None, thresholds[g]))
            keys.append(("none", g, 0.0))
    t0 = time.perf_counter()
    log(f"running {len(tasks)} Monte Carlo runs on {workers} workers ...")
    res = run_parallel(cfg, tasks, workers)
    log(f"done in {time.perf_counter() - t0:.0f} s")

    rows = []
    for kind in ("drift", "jump"):
        for key in sorted({k for k in keys if k[0] == kind}, key=lambda k: (k[1], k[2])):
            pts = [r for r, k in zip(res, keys, strict=True) if k == key]
            rows.append({"attack": kind, "grade": key[1], "level": key[2],
                         **summarize_point(pts)})
    fa = {}
    hours = _truth(cfg).t[-1] / 3600
    for g in grades:
        pts = [r for r, k in zip(res, keys, strict=True) if k == ("none", g, 0.0)]
        n_decl = sum(len(r["declarations"]) for r in pts)
        fa[g] = {"hours": round(len(pts) * hours, 1), "declarations": n_decl,
                 "false_alarms_per_hour": round(n_decl / (len(pts) * hours), 3),
                 "mean_err_protected_m": round(float(np.mean([r["mean_err_p"] for r in pts])), 2),
                 "mean_err_unprotected_m": round(float(np.mean([r["mean_err_u"] for r in pts])),
                                                 2)}
    table = pl.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in row.items()}
                          for row in rows])
    table.write_csv(tables / "b5_sweeps.csv")
    summary = {"runs_per_point": runs, "workers": workers, "calibration": calibration,
               "false_alarms": fa, "sweeps": rows}
    (tables / "b5_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"false alarms: {fa}")
    cols = ["attack", "grade", "level", "p_detect", "delay_median_s", "err_end_unprotected_m",
            "err_end_protected_m", "err_after_protected_m"]
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200):
        log(f"sweeps:\n{table.select(cols)}")
    detection_curve_plot(rows, figs / "b5_detection_curves.png")
    error_reduction_plot(rows, figs / "b5_error_reduction.png")
    log(f"figures in {figs}")
