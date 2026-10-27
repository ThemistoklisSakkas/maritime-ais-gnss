"""B2 - Loosely coupled INS/GNSS error-state (indirect) Extended Kalman Filter, 2D, 8 states.

Nominal state (propagated by the INS at IMU rate):
    p (N, E), v (N, E), psi, b_a (x, y), b_g
Error state (estimated by the filter), all defined as ESTIMATE minus TRUTH:
    dx = [dp_N, dp_E, dv_N, dv_E, dpsi, db_ax, db_ay, db_g]

Error dynamics (linearised about the nominal trajectory; a_n = C(psi) f is the navigation-
frame acceleration, J = [[0, -1], [1, 0]] is a 90 deg rotation):
    d(dp)/dt   = dv
    d(dv)/dt   = J a_n dpsi - C db_a + C n_a       (heading error rotates the acceleration)
    d(dpsi)/dt = -db_g + n_g
    d(db)/dt   = w_b                                (biases as random walks)

Note J a_n: with GNSS position only, a heading error is observable only while the ship
accelerates (turns, speed changes). On a straight, constant-speed leg it is invisible and
its variance grows with the gyro noise: a classic loosely coupled observability result.

GNSS update: z = p_true + e,  innovation nu = z - p_hat,  H = [-I 0 0 0 0] (because
dp = p_hat - p), then the nominal state is corrected by subtracting the estimated error.

Implementation choices
- The nominal state runs at the IMU rate (10 Hz); the covariance is propagated once per
  GNSS epoch (1 Hz) with F built from the interval-mean acceleration and heading. Ship
  dynamics change slowly compared with 1 s, so this costs nothing in accuracy and is 10x
  faster.
- Joseph-form covariance update for numerical robustness.
- Every update returns the innovation, its covariance S and the NIS = nu' S^-1 nu, which
  are the raw material of the integrity monitor in B4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from maritime.nav.sensors import GNSSSpec, IMUSpec

J = np.array([[0.0, -1.0], [1.0, 0.0]])
N_STATES = 8
IDX_P, IDX_V, IDX_PSI, IDX_BA, IDX_BG = slice(0, 2), slice(2, 4), 4, slice(5, 7), 7


@dataclass(frozen=True)
class EKFParams:
    """Filter tuning: what the filter BELIEVES about the sensors."""
    init_sigma_pos_m: float
    init_sigma_vel_mps: float
    init_sigma_heading_deg: float
    accel_noise: float          # white-noise density, m/s^2/sqrt(Hz)
    gyro_noise: float           # rad/s/sqrt(Hz)
    accel_bias_rw: float        # random-walk density of the bias state
    gyro_bias_rw: float
    accel_bias_init: float      # initial bias std (turn-on + instability)
    gyro_bias_init: float
    gnss_sigma_m: float         # per-axis measurement std used in R

    @classmethod
    def from_specs(cls, imu: IMUSpec, gnss: GNSSSpec, d: dict) -> EKFParams:
        """Derive filter noise from the sensor datasheet. A Gauss-Markov bias of std s and
        correlation time T has the same short-term growth as a random walk of density
        sqrt(2 s^2 / T), which is what the filter models."""
        return cls(
            init_sigma_pos_m=float(d["init_sigma_pos_m"]),
            init_sigma_vel_mps=float(d["init_sigma_vel_mps"]),
            init_sigma_heading_deg=float(d["init_sigma_heading_deg"]),
            accel_noise=imu.accel_n, gyro_noise=imu.gyro_n,
            accel_bias_rw=math.sqrt(2 * imu.accel_bi ** 2 / imu.accel_bias_tau_s),
            gyro_bias_rw=math.sqrt(2 * imu.gyro_bi ** 2 / imu.gyro_bias_tau_s),
            accel_bias_init=math.hypot(imu.accel_on, imu.accel_bi),
            gyro_bias_init=math.hypot(imu.gyro_on, imu.gyro_bi),
            gnss_sigma_m=gnss.sigma_total_m * float(d.get("gnss_sigma_scale", 1.0)),
        )

    def initial_covariance(self) -> np.ndarray:
        return np.diag([self.init_sigma_pos_m ** 2] * 2 + [self.init_sigma_vel_mps ** 2] * 2
                       + [math.radians(self.init_sigma_heading_deg) ** 2]
                       + [self.accel_bias_init ** 2] * 2 + [self.gyro_bias_init ** 2])


def rot(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s], [s, c]])


def error_dynamics(a_n: np.ndarray, psi: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time F (8x8) and noise input G (8x6) at the given operating point.

    Noise vector: [n_a (2, body), n_g, w_ba (2), w_bg].
    """
    C = rot(psi)
    F = np.zeros((N_STATES, N_STATES))
    F[IDX_P, IDX_V] = np.eye(2)
    F[IDX_V, IDX_PSI] = J @ a_n
    F[IDX_V, IDX_BA] = -C
    F[IDX_PSI, IDX_BG] = -1.0
    G = np.zeros((N_STATES, 6))
    G[IDX_V, 0:2] = C
    G[IDX_PSI, 2] = 1.0
    G[IDX_BA, 3:5] = np.eye(2)
    G[IDX_BG, 5] = 1.0
    return F, G


def discretize(F: np.ndarray, G: np.ndarray, Qc: np.ndarray, dt: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Phi = I + F dt + (F dt)^2/2;  Qd by the trapezoidal rule (accurate for dt << dynamics)."""
    Fd = F * dt
    Phi = np.eye(len(F)) + Fd + 0.5 * Fd @ Fd
    GQG = G @ Qc @ G.T
    Qd = 0.5 * (Phi @ GQG @ Phi.T + GQG) * dt
    return Phi, Qd


@dataclass
class UpdateResult:
    nu: np.ndarray        # innovation z - z_hat
    S: np.ndarray         # innovation covariance
    nis: float            # normalised innovation squared, ~ chi2(dim z) if all is well
    accepted: bool = True


@dataclass
class InsGnssEKF:
    params: EKFParams
    p: np.ndarray
    v: np.ndarray
    psi: float
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(2))
    b_g: float = 0.0
    P: np.ndarray | None = None
    # accumulators over the current covariance interval
    _sum_a: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _sum_psi: float = 0.0
    _n: int = 0
    _t: float = 0.0

    def __post_init__(self):
        if self.P is None:
            self.P = self.params.initial_covariance()
        q = self.params
        self.Qc = np.diag([q.accel_noise ** 2] * 2 + [q.gyro_noise ** 2]
                          + [q.accel_bias_rw ** 2] * 2 + [q.gyro_bias_rw ** 2])

    # -------------------------------------------------------------- prediction
    def propagate(self, f_meas: np.ndarray, w_meas: float, dt: float) -> None:
        """Nominal INS step with bias-compensated measurements (IMU rate)."""
        w = w_meas - self.b_g
        fx, fy = f_meas[0] - self.b_a[0], f_meas[1] - self.b_a[1]
        psi_new = self.psi + w * dt
        pm = 0.5 * (self.psi + psi_new)
        c, s = math.cos(pm), math.sin(pm)
        a = np.array([c * fx - s * fy, s * fx + c * fy])
        v_new = self.v + a * dt
        self.p = self.p + 0.5 * (self.v + v_new) * dt
        self.v, self.psi = v_new, psi_new
        self._sum_a += a
        self._sum_psi += pm
        self._n += 1
        self._t += dt

    def propagate_covariance(self) -> None:
        """Propagate P over the time accumulated since the last call (GNSS epoch rate)."""
        if self._n == 0:
            return
        F, G = error_dynamics(self._sum_a / self._n, self._sum_psi / self._n)
        Phi, Qd = discretize(F, G, self.Qc, self._t)
        self.P = Phi @ self.P @ Phi.T + Qd
        self._sum_a[:] = 0.0
        self._sum_psi, self._n, self._t = 0.0, 0, 0.0

    # -------------------------------------------------------------- correction
    def innovation(self, z: np.ndarray, R: np.ndarray | None = None) -> UpdateResult:
        """GNSS position innovation and its statistics, without changing the state."""
        H = self.H
        R = self.R if R is None else R
        nu = z - self.p
        S = H @ self.P @ H.T + R
        return UpdateResult(nu=nu, S=S, nis=float(nu @ np.linalg.solve(S, nu)))

    def update(self, z: np.ndarray, R: np.ndarray | None = None) -> UpdateResult:
        H = self.H
        R = self.R if R is None else R
        res = self.innovation(z, R)
        K = self.P @ H.T @ np.linalg.inv(res.S)
        dx = K @ res.nu
        IKH = np.eye(N_STATES) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T           # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        self._correct(dx)
        return res

    def _correct(self, dx: np.ndarray) -> None:
        """Error = estimate - truth, so the corrected estimate is estimate - error."""
        self.p = self.p - dx[IDX_P]
        self.v = self.v - dx[IDX_V]
        self.psi = self.psi - dx[IDX_PSI]
        self.b_a = self.b_a - dx[IDX_BA]
        self.b_g = self.b_g - dx[IDX_BG]

    @property
    def H(self) -> np.ndarray:
        H = np.zeros((2, N_STATES))
        H[:, IDX_P] = -np.eye(2)
        return H

    @property
    def R(self) -> np.ndarray:
        return np.eye(2) * self.params.gnss_sigma_m ** 2


# ---------------------------------------------------------------- run over a scenario

@dataclass
class FilterLog:
    """One row per GNSS epoch, after the update (or after rejection)."""
    t: np.ndarray
    idx: np.ndarray            # IMU-rate index of the epoch
    p: np.ndarray              # (m, 2)
    v: np.ndarray
    psi: np.ndarray
    b_a: np.ndarray            # (m, 2)
    b_g: np.ndarray
    P_diag: np.ndarray         # (m, 8)
    P: np.ndarray              # (m, 8, 8) full covariance, for NEES
    nu: np.ndarray             # (m, 2) innovation
    S: np.ndarray              # (m, 2, 2)
    nis: np.ndarray
    accepted: np.ndarray       # bool


def run_ekf(truth, imu, gnss_t: np.ndarray, gnss_idx: np.ndarray, gnss_p: np.ndarray,
            params: EKFParams, rng: np.random.Generator, monitor=None) -> FilterLog:
    """Run the filter through a scenario.

    Navigation states start from the truth perturbed by their initial uncertainty (so the
    initial error is consistent with P0). Bias estimates start at zero, as on a real
    power-up: the unknown turn-on bias IS the initial error, and P0 describes its spread.
    `monitor`, if given, is called at every GNSS epoch as monitor(ekf, j, z) ->
    (accept: bool, R or None) and decides whether the fix is used: the hook for B4.
    """
    P0 = params.initial_covariance()
    e0 = rng.multivariate_normal(np.zeros(N_STATES), P0)
    ekf = InsGnssEKF(params, p=truth.p[0] + e0[IDX_P], v=truth.v[0] + e0[IDX_V],
                     psi=float(truth.psi[0] + e0[IDX_PSI]), P=P0.copy())
    m = len(gnss_idx)
    log = FilterLog(t=gnss_t, idx=gnss_idx, p=np.empty((m, 2)), v=np.empty((m, 2)),
                    psi=np.empty(m), b_a=np.empty((m, 2)), b_g=np.empty(m),
                    P_diag=np.empty((m, N_STATES)), P=np.empty((m, N_STATES, N_STATES)),
                    nu=np.empty((m, 2)), S=np.empty((m, 2, 2)), nis=np.empty(m),
                    accepted=np.empty(m, dtype=bool))
    dt = truth.dt
    k_prev = 0
    for j, k in enumerate(gnss_idx):
        for i in range(k_prev + 1, k + 1):
            ekf.propagate(imu.f_b[i], imu.w[i], dt)
        k_prev = k
        ekf.propagate_covariance()
        z = gnss_p[j]
        accept, R = (True, None) if monitor is None else monitor(ekf, j, z)
        res = ekf.update(z, R) if accept else ekf.innovation(z, R)
        res.accepted = accept
        log.p[j], log.v[j], log.psi[j] = ekf.p, ekf.v, ekf.psi
        log.b_a[j], log.b_g[j] = ekf.b_a, ekf.b_g
        log.P[j] = ekf.P
        log.P_diag[j] = np.diag(ekf.P)
        log.nu[j], log.S[j], log.nis[j], log.accepted[j] = res.nu, res.S, res.nis, accept
    return log


def error_states(log: FilterLog, truth, imu) -> np.ndarray:
    """Actual error (estimate - truth) at each epoch, in filter state order. (m, 8)."""
    k = log.idx
    dpsi = (log.psi - truth.psi[k] + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack([
        log.p - truth.p[k], log.v - truth.v[k], dpsi,
        log.b_a - imu.accel_bias[k], log.b_g - imu.gyro_bias[k],
    ])


def nees(err: np.ndarray, P: np.ndarray, states: slice | list = slice(None)) -> np.ndarray:
    """Normalised estimation error squared per epoch: e' P^-1 e over the chosen states."""
    e = err[:, states]
    Ps = P[:, states][:, :, states]
    return np.einsum("mi,mi->m", e, np.linalg.solve(Ps, e[..., None])[..., 0])
