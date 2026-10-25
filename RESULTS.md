# Results

Every number produced by the pipeline, with the config and date it came from.
Figures referenced here live in `outputs/figures/`.

## A1 – Ingestion & cleaning

Data: DMA terrestrial AIS, 2025-06-02 .. 2025-06-06 (5 days), AOI 3–17°E, 53.5–58.5°N.
Config: `config/default.yaml` (commit of this section). Full per-day table:
`outputs/tables/a1_cleaning_funnel.csv`.

| Step | Rows removed | % of raw |
|---|---:|---:|
| Raw rows | 94,697,703 | 100.00 % |
| Not a vessel (base stations, AtoN, SAR aircraft) | 6,385,939 | 6.74 % |
| Invalid MMSI (MID outside 201–775) | 639,749 | 0.68 % |
| Invalid position (91/181, 0/0) | 132,528 | 0.14 % |
| Outside AOI | 149,299 | 0.16 % |
| **Exact duplicates (multi-station reception)** | **33,381,531** | **35.25 %** |
| Stale re-stamped repeats (SOG > 1 kn) | 1,017,386 | 1.07 % |
| Low-resolution 1/10′ fixes (msg-27 style) | 27,265 | 0.03 % |
| Isolated position spikes | 1,989 | < 0.01 % |
| **Clean position fixes kept** | **52,962,017** | **55.93 %** |

- Distinct vessels (5 days): **8,950**; ~4,900–5,800 per day.
- Trajectory segments: **43,383** (split on > 30 min silence or impossible jump);
  2,697 segments start at a flagged jump. Median segment duration 0.75 h.
- Cleaning throughput: ~27 s per day (~19 M rows) on an 8 GB laptop, after a 15–20 s
  CSV → Parquet staging step. Peak RAM bounded by per-day processing and 16 MMSI buckets.
- Validation of the jump detector on a real sample: the first version flagged 6 jumps in
  80 k rows, **all false** (stale repeats and coarse msg-27 fixes). Two artefact rules
  reduced that to 0 before the full run.

**Early signal for A4.** Flagged jumps per day were 240–545, except **2025-06-05: 1,294**.
They cluster east of Bornholm (54.8–56.6°N, 14.5–17°E) between **10:00 and 18:00 UTC**,
on up to 12 vessels per hour out of ~150 present, and are absent before and after
(`outputs/figures/a1_bornholm_preview.png`). This is consistent with a regional GNSS
interference event, but MMSI sharing has not been ruled out yet. To be tested in A4.

## A2 – Dark activity
_pending_

## A3 – Ship-to-ship encounters
_pending_

## A4 – GNSS interference
_pending_

## A5 – ETA prediction
_pending_

## B – INS/GNSS resilience (Monte Carlo)
_pending_
