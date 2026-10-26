# Maritime AIS Intelligence & GNSS Resilience

Portfolio project in two parts:

- **Part A – AIS analytics on real data** (Danish Maritime Authority): cleaning, dark activity,
  ship-to-ship encounters, GNSS interference indicators, ETA prediction.
- **Part B – GNSS-resilient navigation** (simulation): INS/GNSS loosely coupled EKF, spoofing and
  jamming injection, innovation-based detection, Monte Carlo evaluation.
- **Part C – Streamlit dashboard** presenting both.

> Work in progress. Results are tracked in [RESULTS.md](RESULTS.md), caveats in
> [LIMITATIONS.md](LIMITATIONS.md).

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[all,dev]"
pytest
```

All parameters live in [config/default.yaml](config/default.yaml). Raw and intermediate data are
written to `data/` (git-ignored); figures and tables to `outputs/`.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Scaffold, config, geo utilities | done |
| A1 | Ingestion, cleaning, trajectories | done |
| A2 | Dark activity | done |
| A3 | Ship-to-ship encounters | done |
| A4 | GNSS interference indicators | done |
| A5 | ETA prediction | done |
| B1 | Trajectory and sensor simulation | done |
| B2 | Loosely coupled EKF | – |
| B3 | Attack injection | – |
| B4 | Innovation-based detection | – |
| B5 | Monte Carlo evaluation | – |
| C | Dashboard | – |
