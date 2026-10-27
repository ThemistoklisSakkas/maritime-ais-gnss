"""B3 - GNSS attack injection on the simulated GNSS position stream.

  jump     spoofing with an abrupt capture: the reported position steps by `magnitude_m`
           in a chosen direction and stays offset for the attack window.
  drift    "meaconing"-style slow pull-off: the offset grows at `rate_mps`, so the spoofed
           track stays smooth and physically plausible. A receiver locked onto a spoofer that
           walks it away gently shows exactly this.
  jamming  denial: no GNSS fixes during the window (reported as NaN), the filter must coast.

The attack modifies MEASUREMENTS only; truth and IMU are untouched. The same scenario and
random seed with and without the attack give a clean comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from maritime.nav.sensors import GNSSData

KINDS = ("jump", "drift", "jamming")


@dataclass(frozen=True)
class Attack:
    kind: str
    start_s: float
    duration_s: float
    direction_deg: float = 90.0     # direction of the spoofed offset (from North, clockwise)
    magnitude_m: float = 0.0        # jump size
    rate_mps: float = 0.0           # drift rate

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown attack kind {self.kind!r}, expected one of {KINDS}")

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def window(self, t: np.ndarray) -> np.ndarray:
        return (t >= self.start_s) & (t < self.end_s)

    def offset(self, t: np.ndarray) -> np.ndarray:
        """Spoofing offset (N, E) added to the true-plus-noise GNSS fix at times t."""
        u = np.array([math.cos(math.radians(self.direction_deg)),
                      math.sin(math.radians(self.direction_deg))])
        if self.kind == "jump":
            mag = np.where(self.window(t), self.magnitude_m, 0.0)
        elif self.kind == "drift":
            mag = np.where(self.window(t), self.rate_mps * (t - self.start_s), 0.0)
        else:
            mag = np.zeros_like(t)
        return mag[:, None] * u[None, :]


def attacks_from_config(d: dict) -> dict[str, Attack]:
    common = {"start_s": float(d["start_s"]), "duration_s": float(d["duration_s"]),
              "direction_deg": float(d["direction_deg"])}
    return {
        "jump": Attack("jump", magnitude_m=float(d["jump"]["magnitude_m"]), **common),
        "drift": Attack("drift", rate_mps=float(d["drift"]["rate_mps"]), **common),
        "jamming": Attack("jamming", **common),
    }


def apply_attack(gnss: GNSSData, attack: Attack | None) -> tuple[GNSSData, np.ndarray]:
    """Return the attacked GNSS stream and a boolean mask of attacked epochs."""
    if attack is None:
        return gnss, np.zeros(len(gnss.t), dtype=bool)
    mask = attack.window(gnss.t)
    p = gnss.p + attack.offset(gnss.t)
    if attack.kind == "jamming":
        p = p.copy()
        p[mask] = np.nan
    return replace(gnss, p=p), mask
