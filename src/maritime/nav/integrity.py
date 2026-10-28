"""B4 - GNSS integrity monitoring for the INS/GNSS EKF: detect, reject, coast, recover.

Two detectors, because B3 showed two very different threats:

1. Per-epoch chi-square gate (catches JUMPS)
   NIS of the main filter > `gate`: the fix is dropped. Isolated outliers are just dropped;
   `n_reject` consecutive rejections declare GNSS untrusted. The filter never absorbed the
   bad fixes, so its state is clean.

2. Backup-INS window test (catches SLOW DRIFT)
   The main filter absorbs a slow drift into its velocity estimate, so its own innovations
   stay small (B3). Every `spawn_s` seconds we therefore take a snapshot of the filter and
   let it run on the IMU ONLY (a backup INS that never sees GNSS). When a backup reaches the
   age `window_s`, the current GNSS fix is compared with it:
        T = r' S^-1 r,  r = z - p_backup,  S = P_backup,pos + R
   A spoofer pulling the GNSS away at rate v separates from a clean backup by ~v * window,
   while the backup's own uncertainty grows only with the residual INS errors. T above
   `window_threshold` declares GNSS untrusted, and the main filter is ROLLED BACK onto that
   backup, discarding everything learned from the spoofed fixes.

While GNSS is untrusted: every fix is rejected, the filter coasts on the INS, optionally
aided by the speed log + non-holonomic constraint (current frozen at its last trusted
estimate). After a `quarantine_s` hold-off, GNSS is re-admitted after `n_reaccept`
consecutive fixes whose test statistic
against the coasting solution is below the (calibrated) window threshold AND that lie within
`reaccept_max_m` of it. The absolute bound keeps a far-off spoofer out even once the
coasting uncertainty has grown large. If GNSS stays out for `max_coast_s` (e.g. a missed slow
drift whose END looked like a jump, so the genuine signal is now the one rejected), the
filter re-initialises its position from GNSS and raises an "integrity not assured" alert.

Thresholds: the chi-square theory does not hold with time-correlated GNSS error (B2), so the
window threshold is calibrated on attack-free Monte Carlo runs (see `calibrate`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from maritime.nav.ekf import InsGnssEKF


@dataclass(frozen=True)
class MonitorParams:
    gate: float                 # per-epoch NIS threshold
    n_reject: int               # consecutive gated fixes that declare GNSS untrusted
    window_s: float             # backup INS age at which it is compared with GNSS
    spawn_s: float              # a new backup every spawn_s seconds
    window_threshold: float     # threshold on T (calibrated)
    n_reaccept: int             # consecutive consistent fixes needed to trust GNSS again
    reaccept_min_m: float       # ... each within max(min, k * sigma_pos) of the coasting
    reaccept_sigma_k: float     #     solution, capped at reaccept_max_m
    reaccept_max_m: float
    quarantine_s: float         # ... and no re-admission at all for this long after a declaration
    max_coast_s: float          # after this long distrusted, re-initialise from GNSS + alert
    arm_after_s: float          # monitor armed only once the filter has converged
    log_aiding: bool
    current_tau_s: float        # smoothing time constant of the current estimate
    sigma_current_mps: float    # uncertainty of the frozen current, enters the log R
    sigma_sway_mps: float       # lateral water speed allowed by the non-holonomic constraint
    log_sigma_mps: float
    log_scale_unc: float        # residual relative scale-factor uncertainty

    @classmethod
    def from_dict(cls, d: dict, window_threshold: float | None = None) -> MonitorParams:
        return cls(
            gate=float(d["gate"]), n_reject=int(d["n_reject"]), window_s=float(d["window_s"]),
            spawn_s=float(d["spawn_s"]),
            window_threshold=float(window_threshold if window_threshold is not None
                                   else d.get("window_threshold", np.inf)),
            n_reaccept=int(d["n_reaccept"]), reaccept_min_m=float(d["reaccept_min_m"]),
            reaccept_sigma_k=float(d["reaccept_sigma_k"]),
            reaccept_max_m=float(d["reaccept_max_m"]),
            quarantine_s=float(d["quarantine_s"]), max_coast_s=float(d["max_coast_s"]),
            arm_after_s=float(d["arm_after_s"]),
            log_aiding=bool(d["log_aiding"]), current_tau_s=float(d["current_tau_s"]),
            sigma_current_mps=float(d["sigma_current_mps"]),
            sigma_sway_mps=float(d["sigma_sway_mps"]), log_sigma_mps=float(d["log_sigma_mps"]),
            log_scale_unc=float(d["log_scale_unc"]),
        )


@dataclass
class Backup:
    ins: InsGnssEKF
    born_t: float
    current: np.ndarray          # current estimate at birth (for a rollback)


@dataclass
class Event:
    t: float
    kind: str                    # "declared" or "reaccepted"
    cause: str                   # "gate", "window" or "consistent"


@dataclass
class IntegrityMonitor:
    params: MonitorParams
    imu: object                  # IMUData
    gnss_t: np.ndarray
    gnss_idx: np.ndarray
    dt: float
    speed_log: np.ndarray | None = None      # log speed at each GNSS epoch
    calibrate: bool = False                  # record statistics only, never declare
    trusted: bool = True
    backups: list = field(default_factory=list)
    events: list = field(default_factory=list)
    window_stats: list = field(default_factory=list)   # (t, T) for every matured backup
    current: np.ndarray | None = None
    _k_prev: int = 0
    _last_spawn: float = -np.inf
    _n_rej: int = 0
    _n_ok: int = 0

    # ------------------------------------------------------------------ main hook
    def __call__(self, ekf: InsGnssEKF, j: int, z: np.ndarray):
        p, t = self.params, float(self.gnss_t[j])
        self._advance_backups(j, ekf)
        u_log = None if self.speed_log is None else float(self.speed_log[j])

        if self.trusted:
            accept = self._trusted_epoch(ekf, t, z)
            if accept and u_log is not None:
                self._update_current(ekf, u_log)
            if self.trusted and t >= p.arm_after_s and t - self._last_spawn >= p.spawn_s:
                self.backups.append(Backup(ekf.snapshot(), t, self._current_or_zero()))
                self._last_spawn = t
            return accept, None

        # GNSS distrusted: coast, aid with the log, test for re-acceptance
        if p.log_aiding and u_log is not None:
            u = float(ekf.v @ [np.cos(ekf.psi), np.sin(ekf.psi)])
            ekf.update_water_speed(
                u_log, self._current_or_zero(),
                sigma_fwd=float(np.sqrt(p.log_sigma_mps ** 2 + (p.log_scale_unc * u) ** 2
                                        + p.sigma_current_mps ** 2)),
                sigma_lat=float(np.hypot(p.sigma_sway_mps, p.sigma_current_mps)))
        # Re-admission compares GNSS with an INS that has been coasting, exactly what the
        # window test does, so it uses the same calibrated threshold (the theoretical gate is
        # too strict for an overconfident coasting covariance), plus an absolute bound that
        # keeps a far-off spoofer out once the coasting uncertainty has grown.
        # Quarantine: a drift is caught while its offset is still small (tens of metres), i.e.
        # inside the re-admission bound. Without a hold-off the monitor would let the spoofer
        # straight back in as soon as the coasting uncertainty grows. If the attack continues,
        # the offset leaves the bound during the quarantine; if it has stopped, genuine GNSS
        # agrees with the coasting solution and is re-admitted.
        # The distance bound follows the coasting uncertainty: an offset the INS can still
        # rule out stays out; one comparable to the INS error cannot be told apart from it.
        #
        # "Consider" term (Schmidt-Kalman idea): the current used by the log aiding is frozen
        # at its pre-attack estimate, so its error is a CONSTANT velocity bias whose position
        # effect grows linearly with coasting time. The filter treats the log error as white
        # noise and averages it away, so its P is far too small (nav grade: sigma 2 m vs 9-34 m
        # actual error after 15 min). We add the ignored term explicitly to the test.
        res = ekf.innovation(z)
        t_coast = t - self._declared_at()
        in_quarantine = t_coast < p.quarantine_s
        consider = (p.sigma_current_mps * t_coast) ** 2 if p.log_aiding else 0.0
        S_eff = res.S + np.eye(2) * consider
        nis_eff = float(res.nu @ np.linalg.solve(S_eff, res.nu))
        sigma = float(np.sqrt(np.linalg.eigvalsh(ekf.P[0:2, 0:2]).max() + consider))
        bound = min(p.reaccept_max_m, max(p.reaccept_min_m, p.reaccept_sigma_k * sigma))
        dist = float(np.linalg.norm(res.nu))
        consistent = not in_quarantine and nis_eff <= p.window_threshold and dist <= bound
        self._n_ok = self._n_ok + 1 if consistent else 0
        if self._n_ok >= p.n_reaccept:
            # Open the position covariance to the observed offset, so the filter converges
            # onto GNSS smoothly instead of the gate seeing the residual as a new attack.
            ekf.P[0:2, 0:2] += np.eye(2) * max(dist ** 2, float(ekf.R[0, 0]))
            self._readmit(t, "consistent")
            return True, None
        # The monitor cannot tell "GNSS is lying" from "my INS has drifted" once both have
        # stayed apart for long (e.g. a missed drift whose end looked like a jump). After
        # max_coast_s, re-initialise the position from GNSS and raise an alert: integrity is
        # no longer assured, a decision for the officer of the watch.
        if t - self._declared_at() >= p.max_coast_s:
            ekf.p = z.copy()
            ekf.P[0:2, :] = 0.0
            ekf.P[:, 0:2] = 0.0
            ekf.P[0:2, 0:2] = ekf.R * 4.0
            ekf.P[2:4, 2:4] += np.eye(2) * 0.25
            self._readmit(t, "timeout_alert")
            return False, None
        return False, None

    def _readmit(self, t: float, cause: str) -> None:
        self.trusted, self._n_ok, self._n_rej = True, 0, 0
        self.backups.clear()
        self._last_spawn = -np.inf
        self.events.append(Event(t, "reaccepted", cause))

    def _declared_at(self) -> float:
        return next(e.t for e in reversed(self.events) if e.kind == "declared")

    # ------------------------------------------------------------------ trusted mode
    def _trusted_epoch(self, ekf: InsGnssEKF, t: float, z: np.ndarray) -> bool:
        p = self.params
        # detector 2 first: a matured clean backup that disagrees with GNSS
        for b in [b for b in self.backups if t - b.born_t >= p.window_s]:
            r = z - b.ins.p
            S = b.ins.P[0:2, 0:2] + b.ins.R
            T = float(r @ np.linalg.solve(S, r))
            self.window_stats.append((t, T))
            self.backups.remove(b)
            if not self.calibrate and T > p.window_threshold:
                ekf.restore(b.ins)                      # roll back to the clean backup
                self.current = b.current.copy()
                self._declare(t, "window")
                return False
        # detector 1: per-epoch gate
        nis = ekf.innovation(z).nis
        if nis > p.gate:
            self._n_rej += 1
            if not self.calibrate and t >= p.arm_after_s and self._n_rej >= p.n_reject:
                self._declare(t, "gate")
            return False
        self._n_rej = 0
        return True

    def _declare(self, t: float, cause: str) -> None:
        self.trusted = False
        self._n_ok = 0
        self.backups.clear()
        self.events.append(Event(t, "declared", cause))

    # ------------------------------------------------------------------ helpers
    def _advance_backups(self, j: int, ekf: InsGnssEKF) -> None:
        """Run every backup INS on the IMU samples up to this epoch (no GNSS, ever).

        Backups fly the same trajectory as the main filter, so over a normal 1-epoch interval
        they reuse its transition matrix (identical to first order, ~5x cheaper). After a gap
        in epochs (e.g. jamming) each backup builds its own.
        """
        k = int(self.gnss_idx[j])
        if self.backups:
            f = self.imu.f_b[self._k_prev + 1:k + 1]
            w = self.imu.w[self._k_prev + 1:k + 1]
            share = abs(len(w) * self.dt - getattr(ekf, "last_dt", -1.0)) < 1e-9
            for b in self.backups:
                b.ins.propagate_block(f, w, self.dt)
                if share:
                    b.ins.apply_transition(ekf.last_Phi, ekf.last_Qd)
                else:
                    b.ins.propagate_covariance()
        self._k_prev = k

    def _update_current(self, ekf: InsGnssEKF, u_log: float) -> None:
        """Water current = ground velocity (filter) - water velocity (log along heading).
        Smoothed; it also soaks up the log's scale-factor error at the current speed."""
        c_now = ekf.v - u_log * np.array([np.cos(ekf.psi), np.sin(ekf.psi)])
        if self.current is None:
            self.current = c_now
        else:
            a = min(1.0, 1.0 / self.params.current_tau_s)    # 1 Hz epochs
            self.current = (1 - a) * self.current + a * c_now

    def _current_or_zero(self) -> np.ndarray:
        return np.zeros(2) if self.current is None else self.current.copy()

    @property
    def first_declaration(self) -> Event | None:
        return next((e for e in self.events if e.kind == "declared"), None)


def calibrate_window_threshold(stats: np.ndarray, pfa: float) -> tuple[float, float]:
    """Fit a SCALED chi-square(2) to attack-free window statistics and return the threshold
    for the per-test false-alarm probability `pfa`, plus the fitted scale.

    With white noise T ~ chi2(2). Correlated GNSS error and the filter's overconfidence
    inflate T by a factor alpha. Fitting the scale (rather than taking an empirical quantile)
    lets us extrapolate to small pfa that a few thousand Monte Carlo tests could never
    resolve directly. The scale is matched at the 99th PERCENTILE, not the mean: false alarms
    live in the tail, and for a better IMU the tail is heavier than a mean-matched fit implies.
    """
    from scipy.stats import chi2

    alpha = float(np.percentile(stats, 99)) / float(chi2.ppf(0.99, 2))
    return alpha * float(chi2.ppf(1.0 - pfa, 2)), alpha
