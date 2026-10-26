"""B1 - Ground-truth ship trajectory in a local North-East frame (flat Earth, 2D).

The ship is described by its speed through the water u and heading psi (from North,
clockwise). A scenario is a list of legs, each with a constant longitudinal acceleration
du/dt and a constant rate of turn r. A uniform water current is added to the ground
velocity, which matters for the speed log (it measures speed through the water).

Body frame: x forward, y starboard. For constant current the specific force seen by a
level IMU is
    f_x = du/dt            (surge acceleration)
    f_y = u * r            (centripetal acceleration in a turn, towards starboard for r > 0)
    w_z = r                (yaw rate)
Gravity, Earth rotation and Schuler dynamics are not modelled (see LIMITATIONS.md).

Truth is integrated on a fine internal step; the IMU-rate outputs are interval MEANS of
f and w, like the delta-v / delta-theta a real IMU delivers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KN = 1852.0 / 3600.0


@dataclass(frozen=True)
class Leg:
    duration_s: float
    accel_mps2: float = 0.0     # du/dt
    turn_rate_dps: float = 0.0  # rate of turn, deg/s (positive = starboard)


@dataclass
class Truth:
    """Arrays at IMU rate. Positions/velocities in the NE frame, angles in radians."""
    t: np.ndarray
    p: np.ndarray        # (n, 2) north, east [m]
    v: np.ndarray        # (n, 2) ground velocity [m/s]
    psi: np.ndarray      # heading [rad]
    u: np.ndarray        # speed through water [m/s]
    f_b: np.ndarray      # (n, 2) mean specific force over the interval ENDING at t [m/s^2]
    w: np.ndarray        # mean yaw rate over the interval ending at t [rad/s]
    current: np.ndarray  # (2,) water current N, E [m/s]
    dt: float


def legs_from_config(items: list[dict]) -> list[Leg]:
    return [Leg(float(d["duration_s"]), float(d.get("accel_mps2", 0.0)),
                float(d.get("turn_rate_dps", 0.0))) for d in items]


def simulate_truth(legs: list[Leg], u0_kn: float, psi0_deg: float, dt: float,
                   substeps: int = 10, current_ne: tuple[float, float] = (0.0, 0.0)) -> Truth:
    """Integrate the ship kinematics leg by leg (midpoint rule on a dt/substeps grid)."""
    h = dt / substeps
    n_total = int(round(sum(leg.duration_s for leg in legs) / dt))
    c = np.asarray(current_ne, dtype=float)

    # per-fine-step accel and turn rate, from the leg schedule
    acc = np.empty(n_total * substeps)
    rate = np.empty(n_total * substeps)
    k = 0
    for leg in legs:
        m = int(round(leg.duration_s / h))
        acc[k:k + m] = leg.accel_mps2
        rate[k:k + m] = np.radians(leg.turn_rate_dps)
        k += m
    acc[k:] = 0.0
    rate[k:] = 0.0

    u = np.empty(n_total * substeps + 1)
    psi = np.empty_like(u)
    u[0], psi[0] = u0_kn * KN, np.radians(psi0_deg)
    u[1:] = u[0] + np.cumsum(acc * h)
    if (u < 0).any():
        raise ValueError("scenario drives the speed through water below zero")
    psi[1:] = psi[0] + np.cumsum(rate * h)

    # midpoint integration of position
    u_mid = 0.5 * (u[1:] + u[:-1])
    psi_mid = 0.5 * (psi[1:] + psi[:-1])
    vel_mid = np.column_stack([u_mid * np.cos(psi_mid), u_mid * np.sin(psi_mid)]) + c
    p = np.vstack([np.zeros(2), np.cumsum(vel_mid * h, axis=0)])

    # specific force on each fine step (midpoint), then mean over each IMU interval
    f_fine = np.column_stack([acc, u_mid * rate])
    f_b = f_fine.reshape(n_total, substeps, 2).mean(axis=1)
    w = rate.reshape(n_total, substeps).mean(axis=1)

    idx = np.arange(0, n_total * substeps + 1, substeps)
    v_all = np.column_stack([u * np.cos(psi), u * np.sin(psi)]) + c
    return Truth(
        t=np.arange(n_total + 1) * dt,
        p=p[idx], v=v_all[idx], psi=psi[idx], u=u[idx],
        # sample k holds the mean over (t_{k-1}, t_k]; sample 0 has no interval -> zeros
        f_b=np.vstack([np.zeros(2), f_b]), w=np.concatenate([[0.0], w]),
        current=c, dt=dt,
    )
