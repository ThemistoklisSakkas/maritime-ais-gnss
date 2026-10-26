import numpy as np
import pytest

from maritime.nav.ins import free_inertial
from maritime.nav.sensors import (
    DPH,
    G0,
    GNSSSpec,
    IMUSpec,
    SpeedLogSpec,
    allan_deviation,
    allan_theory,
    gauss_markov,
    simulate_gnss,
    simulate_imu,
    simulate_speed_log,
)
from maritime.nav.trajectory import KN, Leg, simulate_truth

DT = 0.1
ZERO_IMU = IMUSpec(0, 0, 100, 0, 0, 0, 100, 0)


def scenario():
    """1 h: straight, 90 deg starboard turn, straight, slow down, 180 deg port turn, straight."""
    return [Leg(900), Leg(180, turn_rate_dps=0.5), Leg(600), Leg(120, accel_mps2=-0.0171),
            Leg(360, turn_rate_dps=-0.5), Leg(1440)]


# ---------------------------------------------------------------- truth

def test_straight_leg_distance():
    tr = simulate_truth([Leg(600)], u0_kn=12, psi0_deg=90, dt=DT)
    assert tr.p[-1, 1] == pytest.approx(12 * KN * 600, rel=1e-9)   # due east
    assert abs(tr.p[-1, 0]) < 1e-6


def test_turn_heading_and_centripetal_force():
    tr = simulate_truth([Leg(180, turn_rate_dps=0.5)], u0_kn=12, psi0_deg=0, dt=DT)
    assert np.degrees(tr.psi[-1]) == pytest.approx(90.0)
    r = np.radians(0.5)
    assert np.allclose(tr.f_b[1:, 1], 12 * KN * r)                  # u * r, to starboard
    assert np.allclose(tr.w[1:], r)


def test_current_adds_to_ground_velocity_not_to_specific_force():
    still = simulate_truth([Leg(60)], 10, 0, DT)
    drift = simulate_truth([Leg(60)], 10, 0, DT, current_ne=(0.3, -0.2))
    assert np.allclose(drift.v - still.v, [0.3, -0.2])
    assert np.allclose(drift.f_b, still.f_b)
    assert np.allclose(drift.u, still.u)                            # log speed unchanged


def test_negative_speed_rejected():
    with pytest.raises(ValueError):
        simulate_truth([Leg(100, accel_mps2=-1.0)], 5, 0, DT)


# ---------------------------------------------------------------- error models

def test_perfect_imu_reproduces_truth_inputs():
    tr = simulate_truth(scenario(), 12, 0, DT)
    imu = simulate_imu(tr, ZERO_IMU, np.random.default_rng(0))
    assert np.allclose(imu.f_b, tr.f_b)
    assert np.allclose(imu.w, tr.w)


def test_gauss_markov_steady_state_and_correlation():
    x = gauss_markov(400_000, sigma=2.0, tau=50.0, dt=1.0, rng=np.random.default_rng(1))[:, 0]
    assert x.std() == pytest.approx(2.0, rel=0.05)
    rho = np.corrcoef(x[:-50], x[50:])[0, 1]                        # lag = tau
    assert rho == pytest.approx(np.exp(-1), abs=0.03)


def test_white_noise_level_from_random_walk_coefficient():
    spec = IMUSpec(0, 0, 100, 0.15, 0, 0, 100, 0.03)
    tr = simulate_truth([Leg(36_000)], 0, 0, DT)                    # 10 h at rest
    imu = simulate_imu(tr, spec, np.random.default_rng(2))
    assert imu.w.std() == pytest.approx(spec.gyro_n / np.sqrt(DT), rel=0.01)
    assert imu.f_b[:, 0].std() == pytest.approx(spec.accel_n / np.sqrt(DT), rel=0.01)


def test_allan_deviation_recovers_arw():
    spec = IMUSpec(0, 0, 100, 0.15, 0, 0, 100, 0)                   # ARW only
    tr = simulate_truth([Leg(20_000)], 0, 0, DT)
    imu = simulate_imu(tr, spec, np.random.default_rng(3))
    taus, adev = allan_deviation(imu.w, DT)
    at_1s = np.interp(1.0, taus, adev)
    assert at_1s == pytest.approx(spec.gyro_n, rel=0.05)            # ARW read at tau = 1 s
    slope = np.polyfit(np.log(taus[:10]), np.log(adev[:10]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.03)


def test_allan_deviation_matches_theory_for_gauss_markov():
    spec = IMUSpec(0, 0.5, 300, 0.15, 0, 0, 100, 0)                 # ARW + GM bias
    tr = simulate_truth([Leg(6 * 3600)], 0, 0, DT, substeps=1)
    imu = simulate_imu(tr, spec, np.random.default_rng(6))
    taus, adev = allan_deviation(imu.w - imu.w.mean(), DT)
    theory = allan_theory(taus, spec.gyro_n, spec.gyro_bi, spec.gyro_bias_tau_s)
    # ADEV estimate uncertainty ~ 1/sqrt(2 (T/tau - 1)): with T = 6 h it is ~10 % at
    # tau = 1000 s but ~22 % at 2000 s, so only compare where the estimate is reliable
    sel = (taus >= 1) & (taus <= 1000)
    assert np.allclose(adev[sel], theory[sel], rtol=0.15)


def test_gnss_rate_and_noise():
    spec = GNSSSpec(rate_hz=1, sigma_white_m=2.0, sigma_corr_m=0.0, tau_corr_s=60)
    tr = simulate_truth([Leg(20_000)], 12, 45, DT)
    g = simulate_gnss(tr, spec, np.random.default_rng(4))
    assert np.allclose(np.diff(g.t), 1.0)
    err = g.p - tr.p[g.idx]
    assert err.std(axis=0) == pytest.approx([2.0, 2.0], rel=0.03)
    assert spec.sigma_total_m == pytest.approx(2.0)


def test_speed_log_scale_factor():
    spec = SpeedLogSpec(rate_hz=1, sigma_mps=0.0, scale_factor_ppm=5000)
    tr = simulate_truth([Leg(100)], 10, 0, DT, current_ne=(0.5, 0.0))
    log = simulate_speed_log(tr, spec, np.random.default_rng(5))
    assert np.allclose(log.u, 1.005 * 10 * KN)                      # through water, not SOG


# ---------------------------------------------------------------- INS

def test_ins_with_perfect_imu_follows_truth():
    tr = simulate_truth(scenario(), 12, 30, DT, current_ne=(0.2, 0.1))
    p, v, psi = free_inertial(tr.p[0], tr.v[0], tr.psi[0], tr.f_b, tr.w, DT)
    assert np.abs(psi - tr.psi).max() < 1e-9
    assert np.linalg.norm(p - tr.p, axis=1).max() < 1.0             # < 1 m after 1 h


def test_gyro_bias_gives_linear_heading_error():
    tr = simulate_truth([Leg(600)], 12, 0, DT)
    b = 10 * DPH
    _, _, psi = free_inertial(tr.p[0], tr.v[0], tr.psi[0], tr.f_b, tr.w + b, DT)
    assert psi[-1] - tr.psi[-1] == pytest.approx(b * 600, rel=1e-6)


def test_accel_bias_gives_quadratic_position_error():
    tr = simulate_truth([Leg(300)], 12, 0, DT)
    b = 1e-3 * G0                                                   # 1 mg forward
    p, _, _ = free_inertial(tr.p[0], tr.v[0], tr.psi[0], tr.f_b + [b, 0.0], tr.w, DT)
    assert p[-1, 0] - tr.p[-1, 0] == pytest.approx(0.5 * b * 300 ** 2, rel=1e-3)
