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

## Part B
- The navigation solution is horizontal (2D, 8-state). Roll, pitch and heave are not modelled.
- IMU errors are simulated from a parametric model, not taken from a real sensor log.
