"""B1 - Sensor error models: IMU (2 accelerometers + 1 gyro), GNSS position, speed log.

IMU error per axis, the standard three-term model:

    measured = true + b_on + b_gm(t) + n(t)

  b_on   turn-on bias: constant during a run, different every power-up  ~ N(0, s_on^2)
  b_gm   bias instability as a first-order Gauss-Markov process with correlation time tau
         and steady-state std s_bi:   b[k+1] = e^(-dt/tau) b[k] + s_bi sqrt(1 - e^(-2dt/tau)) w
  n      white noise from the random-walk coefficient (ARW for gyros, VRW for accelerometers):
         a density N [unit/sqrt(Hz)] gives a per-sample std N / sqrt(dt)

Datasheet units are converted here: deg/h, deg/sqrt(h), mg, ug, m/s/sqrt(h).

GNSS position error per axis = white noise + first-order Gauss-Markov (multipath,
atmosphere). The EKF in B2 treats GNSS noise as white, so the correlated part is a
deliberate model mismatch, as in reality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maritime.nav.trajectory import Truth

G0 = 9.80665
DPH = np.pi / 180.0 / 3600.0            # 1 deg/h in rad/s
ARW_UNIT = np.pi / 180.0 / 60.0          # 1 deg/sqrt(h) in rad/sqrt(s)  (sqrt(3600) = 60)
VRW_UNIT = 1.0 / 60.0                    # 1 m/s/sqrt(h) in m/s/sqrt(s)


@dataclass(frozen=True)
class IMUSpec:
    gyro_turn_on_dph: float
    gyro_bias_instability_dph: float
    gyro_bias_tau_s: float
    gyro_arw_dprh: float
    accel_turn_on_mg: float
    accel_bias_instability_ug: float
    accel_bias_tau_s: float
    accel_vrw_mpsprh: float

    @classmethod
    def from_dict(cls, d: dict) -> IMUSpec:
        return cls(**{k: float(v) for k, v in d.items()})

    # SI accessors ----------------------------------------------------------------
    @property
    def gyro_on(self) -> float:
        return self.gyro_turn_on_dph * DPH

    @property
    def gyro_bi(self) -> float:
        return self.gyro_bias_instability_dph * DPH

    @property
    def gyro_n(self) -> float:          # white-noise density, rad/s/sqrt(Hz)
        return self.gyro_arw_dprh * ARW_UNIT

    @property
    def accel_on(self) -> float:
        return self.accel_turn_on_mg * 1e-3 * G0

    @property
    def accel_bi(self) -> float:
        return self.accel_bias_instability_ug * 1e-6 * G0

    @property
    def accel_n(self) -> float:         # m/s^2/sqrt(Hz)
        return self.accel_vrw_mpsprh * VRW_UNIT


@dataclass(frozen=True)
class GNSSSpec:
    rate_hz: float
    sigma_white_m: float
    sigma_corr_m: float
    tau_corr_s: float

    @classmethod
    def from_dict(cls, d: dict) -> GNSSSpec:
        return cls(**{k: float(v) for k, v in d.items()})

    @property
    def sigma_total_m(self) -> float:
        """Per-axis std the filter is told about (white + correlated, root-sum-square)."""
        return float(np.hypot(self.sigma_white_m, self.sigma_corr_m))


@dataclass(frozen=True)
class SpeedLogSpec:
    rate_hz: float
    sigma_mps: float
    scale_factor_ppm: float      # calibration error, parts per million (5000 = 0.5 %)

    @classmethod
    def from_dict(cls, d: dict) -> SpeedLogSpec:
        return cls(**{k: float(v) for k, v in d.items()})


def gauss_markov(n: int, sigma: float, tau: float, dt: float,
                 rng: np.random.Generator, dims: int = 1) -> np.ndarray:
    """First-order Gauss-Markov sequence starting in steady state. Shape (n, dims).

    The recursion x[k] = phi x[k-1] + w[k-1] is a first-order IIR filter, so it runs in
    scipy's compiled lfilter instead of a Python loop.
    """
    from scipy.signal import lfilter

    phi = np.exp(-dt / tau)
    q = sigma * np.sqrt(1.0 - phi ** 2)
    e = np.vstack([rng.normal(0.0, sigma, (1, dims)), rng.normal(0.0, q, (n - 1, dims))])
    return lfilter([1.0], [1.0, -phi], e, axis=0)


@dataclass
class IMUData:
    t: np.ndarray
    f_b: np.ndarray       # measured specific force (n, 2)
    w: np.ndarray         # measured yaw rate (n,)
    accel_bias: np.ndarray  # true total accel bias (n, 2), for analysis only
    gyro_bias: np.ndarray   # true total gyro bias (n,), for analysis only


def simulate_imu(truth: Truth, spec: IMUSpec, rng: np.random.Generator) -> IMUData:
    n, dt = len(truth.t), truth.dt
    a_bias = rng.normal(0.0, spec.accel_on, 2) + gauss_markov(
        n, spec.accel_bi, spec.accel_bias_tau_s, dt, rng, 2)
    g_bias = rng.normal(0.0, spec.gyro_on) + gauss_markov(
        n, spec.gyro_bi, spec.gyro_bias_tau_s, dt, rng, 1)[:, 0]
    a_noise = rng.normal(0.0, spec.accel_n / np.sqrt(dt), (n, 2))
    g_noise = rng.normal(0.0, spec.gyro_n / np.sqrt(dt), n)
    return IMUData(t=truth.t, f_b=truth.f_b + a_bias + a_noise, w=truth.w + g_bias + g_noise,
                   accel_bias=a_bias, gyro_bias=g_bias)


@dataclass
class GNSSData:
    t: np.ndarray
    p: np.ndarray         # measured position (m, 2)
    idx: np.ndarray       # index into the IMU-rate time base


def simulate_gnss(truth: Truth, spec: GNSSSpec, rng: np.random.Generator) -> GNSSData:
    step = int(round(1.0 / (spec.rate_hz * truth.dt)))
    idx = np.arange(step, len(truth.t), step)
    m = len(idx)
    err = rng.normal(0.0, spec.sigma_white_m, (m, 2))
    if spec.sigma_corr_m > 0:
        err += gauss_markov(m, spec.sigma_corr_m, spec.tau_corr_s, 1.0 / spec.rate_hz, rng, 2)
    return GNSSData(t=truth.t[idx], p=truth.p[idx] + err, idx=idx)


@dataclass
class SpeedLogData:
    t: np.ndarray
    u: np.ndarray         # measured speed through water [m/s]
    idx: np.ndarray


def simulate_speed_log(truth: Truth, spec: SpeedLogSpec,
                       rng: np.random.Generator) -> SpeedLogData:
    """Doppler/EM log: speed THROUGH THE WATER with a scale-factor error. With a current,
    it differs from speed over ground: a biased velocity aid unless the current is known."""
    step = int(round(1.0 / (spec.rate_hz * truth.dt)))
    idx = np.arange(step, len(truth.t), step)
    sf = 1.0 + spec.scale_factor_ppm * 1e-6
    return SpeedLogData(t=truth.t[idx],
                        u=sf * truth.u[idx] + rng.normal(0.0, spec.sigma_mps, len(idx)),
                        idx=idx)


# ---------------------------------------------------------------- Allan deviation

def allan_theory(tau: np.ndarray, white_density: float, gm_sigma: float,
                 gm_tau: float) -> np.ndarray:
    """Theoretical Allan deviation of white rate noise + a first-order Gauss-Markov bias.

    White noise:  N^2 / tau.
    Gauss-Markov (std s, correlation time T), standard result:
        (s^2 T^2 / tau^2) * (2 tau / T - 3 + 4 e^(-tau/T) - e^(-2 tau/T))
    It rises, peaks near tau ~ 2T at ~0.62 s, then falls as 1/sqrt(tau): unlike the flat
    flicker floor that datasheets call "bias instability". A GM model only approximates
    that floor over a band of averaging times.
    """
    tau = np.asarray(tau, dtype=float)
    x = tau / gm_tau
    gm = (gm_sigma ** 2 / x ** 2) * (2 * x - 3 + 4 * np.exp(-x) - np.exp(-2 * x))
    return np.sqrt(white_density ** 2 / tau + gm)

def allan_deviation(rate: np.ndarray, dt: float, n_taus: int = 40) -> tuple[np.ndarray,
                                                                          np.ndarray]:
    """Overlapping Allan deviation of a rate signal (e.g. gyro output at rest).

    For white rate noise of density N: sigma(tau) = N / sqrt(tau)  (slope -1/2, read the
    ARW at tau = 1 s). The flat minimum is the bias instability (x ~0.664 for flicker
    noise; for a Gauss-Markov bias the floor sits near its sigma around tau ~ correlation
    time).
    """
    theta = np.concatenate([[0.0], np.cumsum(rate) * dt])   # integrated angle
    n = len(rate)
    ms = np.unique(np.logspace(0, np.log10(n // 3), n_taus).astype(int))
    taus = ms * dt
    adev = np.empty(len(ms))
    for i, m in enumerate(ms):
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        adev[i] = np.sqrt(np.mean(d ** 2) / (2 * (m * dt) ** 2))
    return taus, adev
