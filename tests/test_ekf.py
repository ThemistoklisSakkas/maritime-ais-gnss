import math

import numpy as np
import pytest
from scipy.stats import chi2

from maritime.nav.ekf import (
    IDX_PSI,
    N_STATES,
    EKFParams,
    InsGnssEKF,
    discretize,
    error_dynamics,
    error_states,
    nees,
    run_ekf,
)
from maritime.nav.sensors import (
    DPH,
    G0,
    GNSSSpec,
    IMUSpec,
    simulate_gnss,
    simulate_imu,
)
from maritime.nav.trajectory import Leg, simulate_truth

DT = 0.1
TACTICAL = IMUSpec(3.0, 0.5, 300, 0.15, 0.3, 20, 300, 0.03)
WHITE_GNSS = GNSSSpec(rate_hz=1, sigma_white_m=2.0, sigma_corr_m=0.0, tau_corr_s=60)
INIT = {"init_sigma_pos_m": 3.0, "init_sigma_vel_mps": 0.3, "init_sigma_heading_deg": 1.0}


def scenario():
    return [Leg(300), Leg(180, turn_rate_dps=0.5), Leg(300), Leg(120, accel_mps2=-0.0171),
            Leg(360, turn_rate_dps=-0.5), Leg(540)]


def params(imu=TACTICAL, gnss=WHITE_GNSS):
    return EKFParams.from_specs(imu, gnss, INIT)


# ---------------------------------------------------------------- linearisation

def test_error_dynamics_match_nonlinear_propagation():
    """Propagate a true and a perturbed nominal state through the same IMU samples and
    check that the error evolves as Phi predicts (validates F, including the J a_n term)."""
    rng = np.random.default_rng(0)
    p = params()
    a_true, g_true = np.array([0.02, -0.01]), 5 * DPH
    f_meas = np.array([0.05, 0.08]) + a_true           # accelerating and turning
    w_meas = math.radians(0.4) + g_true
    dx0 = np.array([1.0, -2.0, 0.05, -0.03, math.radians(0.5), 1e-3, -2e-3, 3 * DPH])

    truth = InsGnssEKF(p, p=np.zeros(2), v=np.array([5.0, 3.0]), psi=0.3,
                       b_a=a_true.copy(), b_g=g_true)
    est = InsGnssEKF(p, p=dx0[0:2].copy(), v=truth.v + dx0[2:4], psi=0.3 + dx0[4],
                     b_a=a_true + dx0[5:7], b_g=g_true + dx0[7])
    n = 10
    for _ in range(n):
        truth.propagate(f_meas, w_meas, DT)
        est.propagate(f_meas, w_meas, DT)
    actual = np.concatenate([est.p - truth.p, est.v - truth.v, [est.psi - truth.psi],
                             est.b_a - truth.b_a, [est.b_g - truth.b_g]])
    F, G = error_dynamics(truth._sum_a / n, truth._sum_psi / n)
    Phi, _ = discretize(F, G, np.zeros((6, 6)), n * DT)
    predicted = Phi @ dx0
    assert np.allclose(actual, predicted, rtol=2e-2, atol=1e-4)
    del rng


def test_joseph_update_keeps_covariance_symmetric_positive():
    p = params()
    ekf = InsGnssEKF(p, p=np.zeros(2), v=np.zeros(2), psi=0.0)
    for _ in range(50):
        ekf.propagate(np.zeros(2), 0.0, 1.0)
        ekf.propagate_covariance()
        ekf.update(np.array([0.5, -0.3]))
    assert np.allclose(ekf.P, ekf.P.T)
    assert np.linalg.eigvalsh(ekf.P).min() > 0


# ---------------------------------------------------------------- estimation behaviour

def run_once(seed, imu_spec=TACTICAL, gnss_spec=WHITE_GNSS, filt=None):
    truth = simulate_truth(scenario(), 12, 30, DT)
    rng = np.random.default_rng(seed)
    imu = simulate_imu(truth, imu_spec, rng)
    gnss = simulate_gnss(truth, gnss_spec, rng)
    log = run_ekf(truth, imu, gnss.t, gnss.idx, gnss.p, filt or params(imu_spec, gnss_spec), rng)
    return truth, imu, gnss, log


def test_filter_beats_raw_gnss_and_estimates_biases():
    truth, imu, gnss, log = run_once(1)
    err = error_states(log, truth, imu)
    settled = slice(600, None)                                  # after 10 min
    pos_rms = np.sqrt(np.mean(np.sum(err[settled, 0:2] ** 2, axis=1)))
    raw_rms = np.sqrt(np.mean(np.sum((gnss.p - truth.p[gnss.idx])[settled] ** 2, axis=1)))
    assert pos_rms < 0.8 * raw_rms                              # smoothing helps
    # accelerometer biases: well below the 0.3 mg turn-on level once settled
    assert np.abs(err[settled, 5:7]).mean() < 0.1e-3 * G0
    # every estimate sits within its own 3-sigma most of the time
    inside = np.abs(err) <= 3 * np.sqrt(log.P_diag)
    assert inside[settled].mean() > 0.95


def test_heading_observable_only_when_accelerating():
    _, _, _, log = run_once(2)
    sig = np.sqrt(log.P_diag[:, IDX_PSI])
    # the first turn runs from 300 s to 480 s: heading sigma must drop during it
    assert sig[480] < 0.7 * sig[299]
    # on the long final straight it must grow again (gyro noise, nothing observes it)
    assert sig[-1] > sig[1260]


def test_filter_is_consistent_with_white_gnss_noise():
    """Average NEES over Monte Carlo runs against the chi-square band.

    The filter models the Gauss-Markov biases as random walks and the gyro bias is only
    weakly observable, so it is mildly CONSERVATIVE (NEES a little below the dimension,
    mostly from the gyro-bias state). That is the safe side; what must never happen is
    OVERCONFIDENCE (NEES above the band), which would break the B4 integrity tests.
    """
    runs = 25
    ne = []
    for s in range(runs):
        truth, imu, _, log = run_once(100 + s)
        ne.append(nees(error_states(log, truth, imu), log.P))
    avg = np.mean(ne, axis=0)[60:]                              # skip the first minute
    lo, hi = chi2.ppf([0.025, 0.975], runs * N_STATES) / runs
    assert (avg > hi).mean() < 0.025                            # never overconfident
    assert (avg > lo).mean() > 0.7                              # at most mildly conservative
    assert avg.mean() == pytest.approx(N_STATES, rel=0.2)
