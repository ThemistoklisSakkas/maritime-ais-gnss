import math

import numpy as np
import pytest
from scipy.stats import chi2

from maritime.nav.attacks import Attack, apply_attack
from maritime.nav.ekf import EKFParams, InsGnssEKF, run_ekf
from maritime.nav.integrity import IntegrityMonitor, MonitorParams, calibrate_window_threshold
from maritime.nav.sensors import (
    GNSSSpec,
    IMUSpec,
    SpeedLogSpec,
    simulate_gnss,
    simulate_imu,
    simulate_speed_log,
)
from maritime.nav.trajectory import Leg, simulate_truth

DT = 0.1
TACTICAL = IMUSpec(3.0, 0.5, 300, 0.15, 0.3, 20, 300, 0.03)
GNSS = GNSSSpec(rate_hz=1, sigma_white_m=1.5, sigma_corr_m=1.5, tau_corr_s=60)
LOG = SpeedLogSpec(rate_hz=1, sigma_mps=0.05, scale_factor_ppm=5000)
INIT = {"init_sigma_pos_m": 3.0, "init_sigma_vel_mps": 0.3, "init_sigma_heading_deg": 1.0}
MON = {"gate": 13.8, "n_reject": 3, "window_s": 120, "spawn_s": 30, "n_reaccept": 10,
       "reaccept_max_m": 60, "quarantine_s": 300, "max_coast_s": 600,
       "log_aiding": True, "current_tau_s": 120,
       "sigma_current_mps": 0.05, "sigma_sway_mps": 0.05, "log_sigma_mps": 0.05,
       "log_scale_unc": 0.005}
THRESHOLD = 60.0                     # order of the calibrated value (see B4 results)
ATTACK_START, ATTACK_LEN = 1200, 600


def truth():
    return simulate_truth([Leg(600), Leg(180, turn_rate_dps=0.5), Leg(1500)], 12, 30, DT,
                          current_ne=(0.2, 0.1))


def run(attack, seed=5, protect=True, calibrate=False, **overrides):
    tr = truth()
    rng = np.random.default_rng(seed)
    imu = simulate_imu(tr, TACTICAL, rng)
    gnss, mask = apply_attack(simulate_gnss(tr, GNSS, rng), attack)
    log_u = simulate_speed_log(tr, LOG, rng).u
    mon = None
    if protect:
        mon = IntegrityMonitor(MonitorParams.from_dict({**MON, **overrides}, THRESHOLD),
                               imu, gnss.t, gnss.idx,
                               DT, speed_log=log_u, calibrate=calibrate)
    log = run_ekf(tr, imu, gnss.t, gnss.idx, gnss.p,
                  EKFParams.from_specs(TACTICAL, GNSS, INIT), rng, monitor=mon)
    return mon, np.linalg.norm(log.p - tr.p[log.idx], axis=1), mask, log


# ---------------------------------------------------------------- building blocks

def test_calibration_recovers_scale_of_inflated_chi_square():
    rng = np.random.default_rng(0)
    stats = 3.0 * rng.chisquare(2, 20_000)                     # chi2(2) inflated x3
    thr, alpha = calibrate_window_threshold(stats, 1e-3)
    assert alpha == pytest.approx(3.0, rel=0.03)
    assert thr == pytest.approx(3.0 * chi2.ppf(0.999, 2), rel=0.03)


def test_speed_log_aiding_pulls_velocity_back():
    p = EKFParams.from_specs(TACTICAL, GNSS, INIT)
    psi, u, cur = math.radians(30), 6.0, np.array([0.2, 0.1])
    v_true = u * np.array([math.cos(psi), math.sin(psi)]) + cur
    ekf = InsGnssEKF(p, p=np.zeros(2), v=v_true + [0.4, -0.4], psi=psi)
    err0 = np.linalg.norm(ekf.v - v_true)
    for _ in range(20):
        ekf.update_water_speed(u, cur, sigma_fwd=0.05, sigma_lat=0.05)
    # part of the correction goes into heading (the measurement depends on it), so the
    # velocity error drops by ~10x rather than to the measurement noise
    assert np.linalg.norm(ekf.v - v_true) < err0 / 5


# ---------------------------------------------------------------- end to end

def test_no_attack_no_alarm_and_same_accuracy():
    mon, err, _, log = run(None)
    assert not mon.events
    assert log.accepted.mean() > 0.99
    _, err_raw, _, _ = run(None, protect=False)
    assert np.mean(err[600:]) < 1.2 * np.mean(err_raw[600:])


def test_jump_detected_within_seconds_and_filter_stays_close():
    """Either detector may fire first: the gate after 3 rejected fixes, or a backup INS that
    happens to mature right at the step and sees the full offset at once."""
    att = Attack("jump", ATTACK_START, ATTACK_LEN, 90, magnitude_m=180)
    mon, err, mask, _ = run(att)
    first = mon.first_declaration
    assert first is not None and first.cause in ("gate", "window")
    assert first.t - ATTACK_START <= 5
    assert err[mask].max() < 40                                 # unprotected: > 180 m
    assert any(e.kind == "reaccepted" and e.t > ATTACK_START + ATTACK_LEN for e in mon.events)


def test_slow_drift_detected_by_backup_ins_and_rolled_back():
    att = Attack("drift", ATTACK_START, ATTACK_LEN, 90, rate_mps=0.5)
    mon, err, mask, _ = run(att)
    first = mon.first_declaration
    assert first is not None and first.cause == "window"
    assert first.t - ATTACK_START <= 180                        # window 120 s + spawn 30 s
    # the spoofer must not be let back in while it is still active (quarantine)
    assert not any(e.kind == "reaccepted" and e.t < ATTACK_START + ATTACK_LEN
                   for e in mon.events)
    _, err_raw, _, _ = run(att, protect=False)
    assert err[mask][-1] < 0.3 * err_raw[mask][-1]              # 300 m unprotected
    assert err[-60:].max() < 30                                 # genuine GNSS re-admitted


def test_jamming_needs_no_declaration_and_recovers():
    att = Attack("jamming", ATTACK_START, ATTACK_LEN)
    mon, err, mask, _ = run(att)
    assert not any(e.kind == "declared" for e in mon.events)
    assert err[-120:].max() < 10


def test_calibration_mode_never_acts():
    att = Attack("jump", ATTACK_START, ATTACK_LEN, 90, magnitude_m=180)
    mon, _, _, log = run(att, calibrate=True)
    assert not mon.events and mon.window_stats


def test_missed_drift_then_snap_back_recovers_by_timeout_with_alert():
    """A drift too slow to detect drags the filter away; when the spoofer stops, the true
    GNSS looks like a jump and is rejected. The coasting timeout must re-initialise from
    GNSS (with an alert) instead of rejecting the genuine signal forever."""
    att = Attack("drift", ATTACK_START, ATTACK_LEN, 90, rate_mps=0.1)
    mon, err, mask, _ = run(att, max_coast_s=200)
    assert any(e.kind == "reaccepted" and e.cause == "timeout_alert" for e in mon.events)
    assert err[-60:].max() < 15
