import numpy as np
import pytest

from maritime.nav.attacks import Attack, apply_attack, attacks_from_config
from maritime.nav.ekf import EKFParams, run_ekf
from maritime.nav.sensors import GNSSData, GNSSSpec, IMUSpec, simulate_gnss, simulate_imu
from maritime.nav.trajectory import Leg, simulate_truth

DT = 0.1
TACTICAL = IMUSpec(3.0, 0.5, 300, 0.15, 0.3, 20, 300, 0.03)
GNSS = GNSSSpec(rate_hz=1, sigma_white_m=1.5, sigma_corr_m=1.5, tau_corr_s=60)
INIT = {"init_sigma_pos_m": 3.0, "init_sigma_vel_mps": 0.3, "init_sigma_heading_deg": 1.0}


def clean_stream(n=100):
    t = np.arange(1, n + 1, dtype=float)
    return GNSSData(t=t, p=np.zeros((n, 2)), idx=np.arange(n))


# ---------------------------------------------------------------- injection

def test_jump_offset_only_inside_window():
    g, mask = apply_attack(clean_stream(), Attack("jump", 20, 30, 90, magnitude_m=300))
    assert mask.sum() == 30
    assert np.allclose(g.p[~mask], 0)
    assert np.allclose(g.p[mask], [0.0, 300.0], atol=1e-9)      # due east


def test_drift_grows_linearly_from_zero():
    att = Attack("drift", 20, 60, 0, rate_mps=0.5)               # due north
    g, mask = apply_attack(clean_stream(), att)
    assert g.p[19, 0] == pytest.approx(0.0)                      # t = 20 s, attack begins
    assert g.p[39, 0] == pytest.approx(10.0)                     # 20 s later: 0.5 * 20
    assert np.allclose(g.p[~mask], 0)


def test_jamming_removes_fixes():
    g, mask = apply_attack(clean_stream(), Attack("jamming", 20, 30))
    assert np.isnan(g.p[mask]).all()
    assert not np.isnan(g.p[~mask]).any()


def test_no_attack_is_identity_and_unknown_kind_rejected():
    s = clean_stream()
    g, mask = apply_attack(s, None)
    assert g is s and not mask.any()
    with pytest.raises(ValueError):
        Attack("teleport", 0, 1)


def test_attacks_from_config_share_window():
    a = attacks_from_config({"start_s": 100, "duration_s": 50, "direction_deg": 45,
                             "jump": {"magnitude_m": 300}, "drift": {"rate_mps": 0.5}})
    assert {k: (v.start_s, v.end_s) for k, v in a.items()} == {
        "jump": (100, 150), "drift": (100, 150), "jamming": (100, 150)}


# ---------------------------------------------------------------- filter under attack

def run(attack, seed=3):
    truth = simulate_truth([Leg(600), Leg(180, turn_rate_dps=0.5), Leg(1200)], 12, 30, DT)
    rng = np.random.default_rng(seed)
    imu = simulate_imu(truth, TACTICAL, rng)
    gnss, mask = apply_attack(simulate_gnss(truth, GNSS, rng), attack)
    log = run_ekf(truth, imu, gnss.t, gnss.idx, gnss.p,
                  EKFParams.from_specs(TACTICAL, GNSS, INIT), rng)
    err = np.linalg.norm(log.p - truth.p[log.idx], axis=1)
    return log, err, mask


def test_filter_coasts_through_jamming():
    log, err, mask = run(Attack("jamming", 1200, 300))
    assert not log.available[mask].any() and log.available[~mask].all()
    sig = np.sqrt(log.P_diag[mask, 0])
    assert np.all(np.diff(sig) > 0)                              # uncertainty grows
    assert err[mask].max() < 100                                 # calibrated INS: bounded
    assert err[-60:].max() < 10                                  # re-converges after


def test_unprotected_filter_is_captured_by_jump():
    log, err, mask = run(Attack("jump", 1200, 300, 90, magnitude_m=300))
    assert err[mask][-60:].min() > 250                           # dragged to the fake fix
    assert np.nanmax(log.nis[mask][:3]) > 100                    # but the step was loud


def test_unprotected_filter_follows_slow_drift_quietly():
    """0.2 m/s: the filter is pulled along the full offset while the per-epoch chi-square
    test (99 %, 9.21) fires about as often as it does with no attack at all."""
    log, err, mask = run(Attack("drift", 1200, 300, 90, rate_mps=0.2))
    assert err[mask][-1] > 45                                    # 60 m offset, followed
    assert np.mean(log.nis[mask] > 9.21) < 0.05                  # essentially invisible
