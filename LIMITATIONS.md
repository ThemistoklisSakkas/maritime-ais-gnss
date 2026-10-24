# Limitations

Known caveats, added as each phase is built.

## General
- AIS is a cooperative, self-reported system: positions come from the ship's own GNSS receiver,
  static fields (name, type, destination, ETA) are typed by the crew and are often stale or wrong.
- Terrestrial reception only (DMA shore stations): coverage drops with range from the Danish coast,
  so a "gap" can be a reception artefact rather than a transponder switch-off.

## Part B
- The navigation solution is horizontal (2D, 8-state). Roll, pitch and heave are not modelled.
- IMU errors are simulated from a parametric model, not taken from a real sensor log.
