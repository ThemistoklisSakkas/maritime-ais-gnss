# Limitations

Known caveats, added as each phase is built.

## General
- AIS is a cooperative, self-reported system: positions come from the ship's own GNSS receiver,
  static fields (name, type, destination, ETA) are typed by the crew and are often stale or wrong.
- Terrestrial reception only (DMA shore stations): coverage drops with range from the Danish coast,
  so a "gap" can be a reception artefact rather than a transponder switch-off.

## A1 – Ingestion & cleaning
- Timestamps are the shore base-station receive time (assumed UTC), not the GNSS time of the fix.
  Reception latency is the root cause of "stale" repeated positions; we drop repeats only when
  the vessel reports SOG > 1 kn, so a slow-moving vessel can keep a few stale fixes.
- Low-resolution (1/10') positions are removed by a grid test, not by message type, because the
  DMA CSV does not expose the AIS message ID.
- Isolated-spike removal is a heuristic (A -> X -> B consistency). Two consecutive bad fixes are
  handled by iterating; longer bursts of garbage survive as `jump` flags and segment breaks.
- Spike removal runs per day, so a spike exactly at midnight is not removed (jumps across
  midnight are still flagged when trajectories are built).
- The 50 kn implied-speed limit fits conventional ships; high-speed craft above 50 kn would be
  falsely flagged (rare in Danish waters).

## A2 – Dark activity
- "Coverage" is inferred from the data (other vessels heard in the same 0.1° cell and hour),
  not from a receiver model. A class B transponder (2 W) can drop out where class A (12.5 W)
  ships are still heard, so leisure/fishing gaps are biased towards "dark".
- WPI lists ~144 ports in the AOI but not small marinas or fishing harbours. Most "dark" gaps of
  pleasure and fishing craft are vessels that *stayed in place*: likely moored in an unlisted
  harbour. We therefore report `moved_while_dark` separately and headline commercial types.
- Anchorages are learned from ≥ 3 distinct anchored/moored vessels per cell over the 5 days;
  a rarely used anchorage is missed, and a busy STS hub would be learned as "routine".
- Manoeuvre events at offshore wind farms and construction sites (e.g. Horns Rev, North Sea) are
  routine work, not suspicious; there is no polygon layer for such sites yet.
- Coverage is only checked at the gap's end points. Inland waterways (Kiel Canal) can pass the
  check because the canal entrances are well covered while the middle is not.
- Gaps can only be detected inside the 5-day window; silences that start before or end after it
  are invisible.

## A3 – Ship-to-ship encounters
- The rule finds *proximity*, not cargo transfer. Legitimate operations (bunkering at the Skagen
  anchorage, tugs and pilots alongside, pair trawling, lay-up berths) match it too; the pair
  class and dark-gap context are triage aids, not proof.
- Small harbours missing from WPI (e.g. Skive, Snaptun, Dagebüll) and ships moored together
  produce multi-day "encounters". No coastline test is applied yet (land polygons arrive in A4).
- Distances are between GNSS antennas, not hulls: two ships alongside show ~20–60 m.
- Port proximity is tested at the centre of 0.005° cells (≤ ~300 m error).
- Sailing and pleasure craft are excluded up front; an STS involving a vessel that broadcasts a
  false ship type of that kind would be missed.

## A4 – GNSS interference indicators
- AIS sees interference only through its consequences (impossible jumps, invalid speed,
  erratic course). A receiver that loses lock and simply stops reporting looks like a gap, and
  a small, smooth spoofing offset (< 50 kn implied speed) is invisible here.
- Natural Earth 1:10M closes many Danish fjords and sounds: ~540 k fixes of normal traffic
  fall "deep inland". The land indicator therefore only counts on-land fixes in segments that
  began with an impossible jump. Residual noise remains along the Kiel Canal and Limfjord,
  where reception artefacts occasionally open a segment with a jump.
- Detection cells are 0.5° (~55 × 31 km) and hourly: chosen to match the footprint of a
  jammer; short or very local events are diluted.
- "Affected vessels" counts jumps and on-land fixes, not the cause: an MMSI clash (two ships
  sharing one MMSI) also produces jumps. We checked the two most affected vessels of the
  2025-06-05 event by hand (single name/IMO each, jumps only while transiting the area).
- "Circles" are computed from reported COG. Under degraded GNSS, COG itself is erratic, so a
  circle can be noise rather than a real circular track; outside interference periods most
  detected circles are ferries, HSC and survey vessels.
- No ground truth: there is no public, time-stamped list of jamming incidents to validate
  against, so results are indicators, not confirmed events.

## A5 – ETA prediction
- Only 5 days of data: voyages longer than the window are truncated, so short coastal trips
  dominate; horizons are capped at 48 h. The test set is ~1.5 days of later arrivals.
- "Arrival" = first fix inside a WPI port radius followed by a ≥ 1 h mostly-stopped stay. It is
  not the berthing time, and lock waits (Kiel Canal entrances) count as arrivals.
- **Waiting at anchor outside the port** (for a berth or pilot) is the dominant error at long
  horizons: the vessel is close, so every kinematic method predicts a few hours while the true
  wait is a day or more. Fixing this needs port congestion / berth schedule data, not AIS.
- The destination is assumed known (the port actually reached). A deployed system would have
  to infer or look it up.
- Great-circle distance ignores land: routes around Jutland or through the Belts are longer.
  The model partly learns this from position features; a proper route network (e.g. a
  sea-lane graph) would do better.
- AIS-reported ETA ("crew ETA" in code): this is the manually entered msg-5 field, not the
  ECDIS navigational ETA, which AIS does not carry. Some integrated bridges can feed it
  automatically; we cannot tell which vessels do. It is matched to the arrival port by
  UN/LOCODE or port-name substring; exonyms ("GOTHENBURG" vs WPI "Goteborg") are missed.
  The AIS ETA has no year; we re-attach the year closest to the report time. 22 % of matched
  values were already in the past (not updated), so results are reported both raw and on a
  "plausible" subset.
- The baseline uses great-circle distance; a real ECDIS ETA uses the passage-plan route and
  would be more accurate at long horizons than our baseline.
- The model target transform (log hours) was chosen on a validation split inside the
  training period; the test set was not used for any choice. On test, the baseline remains
  slightly better in the last 3 h (0.93 vs 1.10 h MAE).

## Part B
- The navigation solution is horizontal (2D, 8-state). Roll, pitch and heave are not modelled.
- IMU errors are simulated from a parametric model, not taken from a real sensor log.
- **Flat, non-rotating Earth, no gravity coupling:** there is no Schuler loop. In a real
  INS a horizontal accelerometer bias produces a position error bounded by the 84-min
  Schuler oscillation (amplitude ~ b R / g); here it grows as ½ b t² without bound. The two
  agree for t ≪ T_S/4 ≈ 21 min, which covers the spoofing/jamming windows studied, but the
  1-hour free-inertial numbers overstate real drift. Earth rate (12°/h vertical at 55° N) is
  also absent, which flatters low-grade gyros.
- "Bias instability" is modelled as a first-order Gauss-Markov process. Datasheet bias
  instability is the flat flicker floor of the Allan deviation; a GM process only
  approximates it over a band of averaging times (see `b1_allan_deviation.png`).
- Ship motion has no sway/drift angle and no wave-induced accelerations; turns are
  constant-rate. Real ship IMUs see much richer, noisier dynamics.
- GNSS is simulated as position only with white + Gauss-Markov errors; no satellite
  geometry (DOP), no outliers, no receiver clock or velocity outputs.
- **The EKF assumes white GNSS noise.** With realistic time-correlated GNSS error
  (τ = 60 s) it becomes strongly overconfident: Monte Carlo NEES ~75 vs 8, and the
  estimate follows the slow GNSS error instead of averaging it out. Inflating R (×1.5 σ)
  only halves the NEES. The principled fix is to augment the state with two Gauss-Markov
  GNSS-error states (10-state filter) or to use a Schmidt-Kalman "consider" formulation; not
  done here to keep the agreed 8-state design. Consequence for B4: chi-square thresholds
  derived for white noise do not hold, so detection thresholds are calibrated empirically
  on attack-free Monte Carlo runs.
- NIS cannot reveal this mismatch: with correlated GNSS error the average NIS is *below* its
  expected value (1.4 vs 2) while NEES is ~10× too high. Innovation monitoring sees only
  what disagrees with the filter's own prediction — the same blind spot a slow spoofer uses.
- **Attack model (B3)** is a simple offset on the position output. A real spoofer must first
  capture the receiver (lift-off, which may itself cause signal-quality anomalies), and it
  also shifts velocity and clock; jamming in reality often degrades accuracy before losing
  lock. We do not model receiver tracking loops, C/N0, AGC or multi-constellation checks,
  which are real-world detection layers in addition to the INS-based test studied here.
- The attacker here does not try to match the ship's own manoeuvres or to stay under a
  known detector threshold; an adaptive attacker is harder to detect than these scenarios.
- Loosely coupled, position-only: heading is unobservable on straight constant-speed legs
  (no gyrocompass or GNSS-heading aiding modelled), so its uncertainty grows between turns.
