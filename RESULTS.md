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

Threshold: silence ≥ 2 h. Zones: 144 WPI ports (radius by harbour size) + 418 learned
anchorage cells (≥ 3 distinct anchored/moored vessels). Coverage: ≥ 3 other vessels heard in
the same 0.1° cell during ≥ 80 % of the gap's interior hours, at both ends.
Full output: `outputs/tables/a2_summary.json`, `outputs/tables/a2_dark_gaps_top.csv`.
Runtime: 23 s for 53 M fixes (after A1).

| Explanation assigned | Gaps | % |
|---|---:|---:|
| Port or anchorage | 6,580 | 54.4 % |
| Low receiver coverage | 3,246 | 26.8 % |
| Position jump (> 30 kn to bridge the gap) | 1 | < 0.1 % |
| **Unexplained ("dark")** | **2,268** | **18.8 %** |
| Total gaps ≥ 2 h | 12,095 | |

- Dark gaps come from 1,111 vessels, but 1,827 of the 2,268 (81 %) are **sailing and
  pleasure craft**, and 2,124 (94 %) **stayed in place**: most likely small craft in marinas
  that WPI does not list. Only **144** dark gaps show the vessel **moving ≥ 1 NM while silent**.
- Commercial traffic (Cargo/Tanker): **12 dark gaps from 10 vessels** in 5 days; 7 of them were
  stationary (SOG 0 before and after), **5 moved while dark**.
- Manual triage of the 5 commercial "moved while dark" gaps:
  - 2 inside the **Kiel Canal** (54.2–54.4°N, 9.4–9.9°E) at 5–6 kn, the canal speed:
    inland transit with poor Danish reception, likely benign → the coverage proxy failed here.
  - 1 in the **Gothenburg approaches**, 2 NM in 2.8 h after slowing from 10 kn: waiting
    just outside the WPI port radius, likely benign.
  - **2 unexplained**: one continued 56 NM at its previous speed overnight between the
    Øresund and Bornholm (good coverage at both ends); one **is the same MMSI that produced
    358 position jumps east of Bornholm on 2025-06-05** (A1) — carried into A4.
- Abrupt manoeuvres: 15,414 events in total, 9,093 outside ports/anchorages (4,469 U-turns,
  4,624 abrupt stops); 278 by Cargo/Tanker. These cluster at offshore wind-farm and
  construction sites (e.g. 55.6°N 7.6°E), i.e. mostly routine work.

**Interview takeaway:** an automatic rule cut 12,095 gaps to 5 commercial candidates; human
review of 5 cases took minutes and left 2. The detector is a triage tool, not a verdict.

## A3 – Ship-to-ship encounters

Rule: two vessels < 500 m apart, both < 3 kn, ≥ 1 h, outside ports/anchorages; sailing and
pleasure craft excluded. 5-min bins, up to 10 min reception dropout bridged.
Output: `outputs/tables/a3_summary.json`, `outputs/tables/a3_encounters_cargo_tanker.csv`.

| Funnel | Vessel-bins |
|---|---:|
| Slow (< 3 kn) vessel positions, 5-min bins | 3,353,302 |
| − inside ports / learned anchorages | 2,229,444 |
| − sailing / pleasure craft | 639,493 |
| **Candidates for pairing** | **484,365** |

- **Spatial indexing:** 1,440 time bins, up to 421 candidates per bin. A brute-force search
  would need **81.9 M** pairwise distances; the per-bin KD-tree returns the 242,968 close
  pairs in **1.4 s** (whole A3 stage 7.5 s). A unit test proves the KD-tree result equals
  the brute-force result on random data. The gap widens quadratically with traffic density.
- **Encounters found: 2,534** (median 3.1 h): 1,351 with a tug/pilot/SAR/other service
  vessel, 727 fishing pairs, 441 other, **15 cargo/tanker pairs**.
- 696 encounters have a dark gap (A2) of either vessel within ±12 h: 392 fishing, 212
  service, 92 other, **0 cargo/tanker**.
- Manual triage of the 15 cargo/tanker encounters:
  - **5 at the Skagen anchorage** (57.5–57.7°N, 10.6°E): a tanker alongside another ship
    for 3–9 h at 20–70 m antenna separation — consistent with routine **bunkering**, a true
    STS operation of a legitimate kind (`outputs/figures/a3_encounter_example.png`).
  - ~8 are ships moored together in **small harbours missing from WPI** (Skive, Snaptun,
    Dagebüll, one berth on west Zealand seen 5 times in 5 days) → false positives.
  - The rest need a coastline test to judge (planned with land polygons in A4).
- The pipeline is deterministic: the development run and the user's run produced identical
  counts.

## A4 – GNSS interference
_pending_

## A5 – ETA prediction
_pending_

## B – INS/GNSS resilience (Monte Carlo)
_pending_
