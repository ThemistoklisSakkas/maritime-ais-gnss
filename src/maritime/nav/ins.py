"""B1/B2 - 2D strapdown INS mechanization in the local North-East frame.

State: position p (N, E), ground velocity v (N, E), heading psi. Inputs per IMU step:
specific force f_b (x forward, y starboard) and yaw rate w, both as interval means.

    psi_k   = psi_{k-1} + w_k dt
    a_n     = C(psi_mid) f_b,k          C = [[cos, -sin], [sin, cos]]
    v_k     = v_{k-1} + a_n dt
    p_k     = p_{k-1} + (v_{k-1} + v_k) / 2 dt

Using the mid-interval heading for the body-to-nav rotation makes the scheme second-order
accurate, matching how the truth was generated.
"""

from __future__ import annotations

import numpy as np


def rot(psi: float | np.ndarray) -> np.ndarray:
    """Body (x fwd, y stbd) -> nav (N, E) rotation for heading psi (rad, clockwise from N)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def mechanize_step(p: np.ndarray, v: np.ndarray, psi: float, f_b: np.ndarray, w: float,
                   dt: float) -> tuple[np.ndarray, np.ndarray, float]:
    psi_new = psi + w * dt
    a_n = rot(0.5 * (psi + psi_new)) @ f_b
    v_new = v + a_n * dt
    p_new = p + 0.5 * (v + v_new) * dt
    return p_new, v_new, psi_new


def free_inertial(p0: np.ndarray, v0: np.ndarray, psi0: float, f_b: np.ndarray,
                  w: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unaided navigation from a known initial state. Sample 0 is the initial state."""
    p, v, psi = free_inertial_batch(p0, v0, psi0, f_b[None], w[None], dt)
    return p[0], v[0], psi[0]


def free_inertial_batch(p0: np.ndarray, v0: np.ndarray, psi0: float, f_b: np.ndarray,
                        w: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same mechanization for R Monte Carlo runs at once: f_b (R, n, 2), w (R, n).

    Unaided, each state is a running sum of the previous one, so the whole mechanization is
    three cumulative sums over time (heading, then velocity, then position): no Python loop.
    The aided filter (B2) cannot do this, because every GNSS update resets the state.
    """
    r, n = w.shape
    psi = psi0 + np.cumsum(np.concatenate([np.zeros((r, 1)), w[:, 1:] * dt], axis=1), axis=1)
    psi_mid = 0.5 * (psi[:, 1:] + psi[:, :-1])
    c, s = np.cos(psi_mid), np.sin(psi_mid)
    fx, fy = f_b[:, 1:, 0], f_b[:, 1:, 1]
    a_n = np.stack([c * fx - s * fy, s * fx + c * fy], axis=-1)        # (R, n-1, 2)
    v = np.concatenate([np.broadcast_to(v0, (r, 1, 2)),
                        v0 + np.cumsum(a_n * dt, axis=1)], axis=1)
    p = np.concatenate([np.broadcast_to(p0, (r, 1, 2)),
                        p0 + np.cumsum(0.5 * (v[:, 1:] + v[:, :-1]) * dt, axis=1)], axis=1)
    return p, v, psi
