"""Part B orchestration, one `run_bX` function per phase."""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from maritime.config import Config
from maritime.nav.attacks import Attack, apply_attack, attacks_from_config
from maritime.nav.ekf import EKFParams, error_states, nees, run_ekf
from maritime.nav.ins import free_inertial_batch
from maritime.nav.integrity import (
    IntegrityMonitor,
    MonitorParams,
    calibrate_window_threshold,
)
from maritime.nav.sensors import (
    GNSSSpec,
    IMUSpec,
    SpeedLogSpec,
    allan_deviation,
    simulate_gnss,
    simulate_imu,
    simulate_speed_log,
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


# ---------------------------------------------------------------- B3

def attack_mc(truth: Truth, imu_spec: IMUSpec, gnss_spec: GNSSSpec, params: EKFParams,
              attack: Attack | None, runs: int, seed: int, monitor_factory=None) -> dict:
    """Monte Carlo of one attack. Same seeds for every attack -> paired comparisons.

    monitor_factory(truth, imu, gnss, rng) must return a fresh monitor per run (B4);
    None = unprotected filter.
    Returns per-run horizontal error, NIS, accepted/available flags and the attack mask.
    """
    out = {"err": [], "nis": [], "accepted": [], "available": [], "gnss_err": [],
           "events": []}
    for r in range(runs):
        rng = np.random.default_rng(seed + r)
        imu = simulate_imu(truth, imu_spec, rng)
        gnss, mask = apply_attack(simulate_gnss(truth, gnss_spec, rng), attack)
        mon = None
        if monitor_factory:
            # the speed log has its own random stream, so protected and unprotected runs of
            # the same seed see identical IMU, GNSS and initial errors (paired comparison)
            mon = monitor_factory(truth, imu, gnss, np.random.default_rng([seed + r, 1]))
        log = run_ekf(truth, imu, gnss.t, gnss.idx, gnss.p, params, rng, monitor=mon)
        out["events"].append(list(mon.events) if mon else [])
        out["err"].append(np.linalg.norm(log.p - truth.p[log.idx], axis=1))
        out["nis"].append(log.nis)
        out["accepted"].append(log.accepted)
        out["available"].append(log.available)
        out["gnss_err"].append(np.linalg.norm(gnss.p - truth.p[gnss.idx], axis=1))
        if r == 0:
            out["log"], out["monitor"] = log, mon
    for k in ("err", "nis", "accepted", "available", "gnss_err"):
        out[k] = np.stack(out[k])
    out["mask"], out["t"] = mask, log.t
    return out


def attack_metrics(mc: dict, attack: Attack | None, gate: float, recovery_s: int = 300) -> dict:
    t, mask, err = mc["t"], mc["mask"], mc["err"]
    if attack is None:
        win = (t >= 600)
        return {"mean_err_m": round(float(err[:, win].mean()), 2),
                "p95_err_m": round(float(np.percentile(err[:, win], 95)), 2),
                "nis_over_gate_pct": round(100 * float(np.nanmean(mc["nis"][:, win] > gate)), 2)}
    after = (t >= attack.end_s) & (t < attack.end_s + recovery_s)
    end_i = int(np.flatnonzero(mask)[-1])
    nis_att = mc["nis"][:, mask]
    return {
        "max_err_during_median_m": round(float(np.median(err[:, mask].max(axis=1))), 1),
        "max_err_during_p95_m": round(float(np.percentile(err[:, mask].max(axis=1), 95)), 1),
        "err_at_attack_end_median_m": round(float(np.median(err[:, end_i])), 1),
        "mean_err_5min_after_m": round(float(err[:, after].mean()), 1),
        "nis_over_gate_pct": (None if np.isnan(nis_att).all() else
                              round(100 * float(np.nanmean(nis_att > gate)), 1)),
    }


def run_b3(cfg: Config) -> None:
    from maritime.plots import attack_overview_plot, drift_stealth_plot

    s, b3 = cfg.section("sim"), cfg.section("b3")
    tables, figs = cfg.paths.tables, cfg.paths.figures
    cfg.paths.ensure()
    truth = scenario_truth(cfg)
    imu_spec = IMUSpec.from_dict(s["imu_grades"][s["imu_grade"]])
    gnss_spec = GNSSSpec.from_dict(s["gnss"])
    params = ekf_params(cfg, imu_spec, gnss_spec)
    attacks = attacks_from_config(cfg.section("attacks"))
    gate, runs = float(b3["chi2_gate"]), int(b3["mc_runs"])

    mcs, metrics = {}, {}
    for name, att in [("none", None), *attacks.items()]:
        t0 = time.perf_counter()
        mcs[name] = attack_mc(truth, imu_spec, gnss_spec, params, att, runs, cfg.seed)
        metrics[name] = attack_metrics(mcs[name], att, gate)
        _log(f"{name}: {runs} runs in {time.perf_counter() - t0:.0f} s -> {metrics[name]}")

    # how stealthy is a slow drift? per-epoch detectability vs rate
    sweep = []
    base = attacks["drift"]
    for rate in b3["drift_sweep_mps"]:
        att = Attack("drift", base.start_s, base.duration_s, base.direction_deg, rate_mps=rate)
        mc = attack_mc(truth, imu_spec, gnss_spec, params, att, int(b3["drift_sweep_runs"]),
                       cfg.seed + 1000)
        m = attack_metrics(mc, att, gate)
        sweep.append({"rate_mps": rate, "offset_at_end_m": rate * att.duration_s,
                      "nis_over_gate_pct": m["nis_over_gate_pct"],
                      "filter_err_at_end_m": m["err_at_attack_end_median_m"]})
        _log(f"drift sweep {rate} m/s -> {sweep[-1]}")
    sweep = pl.DataFrame(sweep)
    sweep.write_csv(tables / "b3_drift_stealth.csv")

    # jamming reference: the same outage length with an UNCALIBRATED INS (B1 free inertial)
    fi = free_inertial_drift(truth, imu_spec, 200, cfg.seed + 2000)
    fi_end = float(np.median(fi[:, int(round(base.duration_s / truth.dt))]))

    summary = {"runs": runs, "chi2_gate": gate,
               "attack_window_s": [base.start_s, base.end_s], "unprotected": metrics,
               "jamming_vs_uncalibrated_free_inertial_m": {
                   "ekf_calibrated_median": metrics["jamming"]["err_at_attack_end_median_m"],
                   "free_inertial_median": round(fi_end, 1)},
               "drift_sweep": sweep.to_dicts()}
    (tables / "b3_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"summary: {json.dumps(summary, indent=2)}")
    attack_overview_plot(mcs, attacks, gate, figs / "b3_attacks_unprotected.png")
    drift_stealth_plot(sweep, metrics["none"]["nis_over_gate_pct"],
                       figs / "b3_drift_stealth.png")
    _log(f"figures in {figs}")


# ---------------------------------------------------------------- B4

def make_monitor_factory(cfg: Config, threshold: float, calibrate: bool = False):
    """Factory building a fresh IntegrityMonitor (and its speed log) for each run."""
    mp = MonitorParams.from_dict(cfg.section("integrity"), threshold)
    log_spec = SpeedLogSpec.from_dict(cfg.section("sim")["speed_log"])

    def factory(truth: Truth, imu, gnss, rng):
        log_u = simulate_speed_log(truth, log_spec, rng).u
        return IntegrityMonitor(mp, imu, gnss.t, gnss.idx, truth.dt, speed_log=log_u,
                                calibrate=calibrate)
    return factory


def calibrate_monitor(cfg: Config, truth: Truth, imu_spec: IMUSpec, gnss_spec: GNSSSpec,
                      params: EKFParams) -> dict:
    """Window-test threshold from attack-free runs (independent seeds), scaled chi-square."""
    b4, ic = cfg.section("b4"), cfg.section("integrity")
    stats = []
    for r in range(int(b4["calib_runs"])):
        seed = cfg.seed + int(b4["calib_seed_offset"]) + r
        rng = np.random.default_rng(seed)
        imu = simulate_imu(truth, imu_spec, rng)
        gnss = simulate_gnss(truth, gnss_spec, rng)
        mon = make_monitor_factory(cfg, np.inf, calibrate=True)(
            truth, imu, gnss, np.random.default_rng([seed, 1]))
        run_ekf(truth, imu, gnss.t, gnss.idx, gnss.p, params, rng, monitor=mon)
        stats += [T for t, T in mon.window_stats if t >= float(b4["calib_skip_s"])]
    stats = np.asarray(stats)
    thr, alpha = calibrate_window_threshold(stats, float(ic["pfa_window"]))
    return {"threshold": thr, "alpha": alpha, "n_tests": int(stats.size),
            "mean_T": float(stats.mean()), "p99_T": float(np.percentile(stats, 99)),
            "max_T": float(stats.max()),
            "calib_exceed_pct": float(100 * np.mean(stats > thr)), "stats": stats}


def detection_metrics(mc: dict, attack: Attack | None, hours_per_run: float) -> dict:
    """First declaration per run -> detection rate, delay, false alarms, cause."""
    firsts = [next((e for e in ev if e.kind == "declared"), None) for ev in mc["events"]]
    if attack is None:
        n_decl = sum(sum(e.kind == "declared" for e in ev) for ev in mc["events"])
        return {"runs_with_false_alarm": sum(f is not None for f in firsts),
                "false_alarms_per_hour": round(n_decl / (len(firsts) * hours_per_run), 3)}
    early = [f for f in firsts if f is not None and f.t < attack.start_s]
    hits = [f for f in firsts if f is not None and f.t >= attack.start_s]
    during = [f for f in hits if f.t < attack.end_s]
    delays = [f.t - attack.start_s for f in during]
    causes = {c: sum(f.cause == c for f in during) for c in ("gate", "window")}
    return {
        "detected_during_attack_runs": f"{len(during)}/{len(firsts)}",
        "detected_only_at_attack_end_runs": len(hits) - len(during),
        "detection_delay_median_s": float(np.median(delays)) if delays else None,
        "detection_delay_max_s": float(np.max(delays)) if delays else None,
        "detected_by": causes,
        "false_alarm_before_attack_runs": len(early),
        "runs_with_timeout_alert": sum(any(e.cause == "timeout_alert" for e in ev)
                                       for ev in mc["events"]),
        **_split_by_detection(mc, attack, firsts),
    }


def _split_by_detection(mc: dict, attack: Attack, firsts: list) -> dict:
    """Protected-filter error at the end of the attack, separately for runs where the attack
    was detected while active and runs where it was missed (medians would hide the split)."""
    end_i = int(np.flatnonzero(mc["mask"])[-1])
    det = np.array([f is not None and attack.start_s <= f.t < attack.end_s for f in firsts])
    err_end = mc["err"][:, end_i]
    after = (mc["t"] >= attack.end_s) & (mc["t"] < attack.end_s + 300)
    err_after = mc["err"][:, after].mean(axis=1)

    def med(x):
        return round(float(np.median(x)), 1) if x.size else None
    return {"err_at_end_detected_runs_m": med(err_end[det]),
            "err_at_end_missed_runs_m": med(err_end[~det]),
            "err_5min_after_detected_runs_m": med(err_after[det]),
            "err_5min_after_missed_runs_m": med(err_after[~det])}


def run_b4(cfg: Config) -> None:
    from maritime.plots import protection_comparison_plot, window_calibration_plot

    s, b4 = cfg.section("sim"), cfg.section("b4")
    tables, figs = cfg.paths.tables, cfg.paths.figures
    cfg.paths.ensure()
    truth = scenario_truth(cfg)
    imu_spec = IMUSpec.from_dict(s["imu_grades"][s["imu_grade"]])
    gnss_spec = GNSSSpec.from_dict(s["gnss"])
    params = ekf_params(cfg, imu_spec, gnss_spec)
    attacks = attacks_from_config(cfg.section("attacks"))
    runs = int(b4["mc_runs"])

    t0 = time.perf_counter()
    cal = calibrate_monitor(cfg, truth, imu_spec, gnss_spec, params)
    _log(f"calibration ({b4['calib_runs']} attack-free runs, {cal['n_tests']} window tests, "
         f"{time.perf_counter() - t0:.0f} s): mean T {cal['mean_T']:.2f} (chi2 theory 2), "
         f"alpha {cal['alpha']:.2f} -> threshold {cal['threshold']:.1f}")
    factory = make_monitor_factory(cfg, cal["threshold"])

    results, mcs_u, mcs_p = {}, {}, {}
    hours = truth.t[-1] / 3600
    for name, att in [("none", None), *attacks.items()]:
        t0 = time.perf_counter()
        mcs_u[name] = attack_mc(truth, imu_spec, gnss_spec, params, att, runs, cfg.seed)
        mcs_p[name] = attack_mc(truth, imu_spec, gnss_spec, params, att, runs, cfg.seed,
                                monitor_factory=factory)
        results[name] = {
            "unprotected": attack_metrics(mcs_u[name], att, params_gate(cfg)),
            "protected": attack_metrics(mcs_p[name], att, params_gate(cfg)),
            "detection": detection_metrics(mcs_p[name], att, hours),
        }
        _log(f"{name} ({time.perf_counter() - t0:.0f} s): {results[name]}")

    summary = {"runs_per_case": runs,
               "calibration": {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in cal.items() if k != "stats"},
               "cases": results}
    (tables / "b4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"summary: {json.dumps(summary, indent=2, default=str)}")
    protection_comparison_plot(mcs_u, mcs_p, attacks, figs / "b4_protected_vs_unprotected.png")
    window_calibration_plot(cal, figs / "b4_window_test_calibration.png")
    _log(f"figures in {figs}")


def params_gate(cfg: Config) -> float:
    return float(cfg.section("integrity")["gate"])
