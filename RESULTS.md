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

Indicators per 0.5° cell (~55 × 31 km) and hour, counted as distinct vessels: impossible
jumps (A1) and fixes > 1 km inside Natural Earth land in a segment that began with a jump.
Flag: ≥ 3 affected vessels and ≥ 5 % of vessels present. Spoofing test: ≥ 3 vessels jumping
into the same 0.05° cell in the same hour. Scan of 53 M fixes: 68 s.
Output: `outputs/tables/a4_summary.json`, `outputs/tables/a4_hourly_bornholm_vs_rest.csv`,
figures `a4_interference_daily.png`, `a4_interference_timeseries.png`.

| Day | Flagged cell-hours | Affected vessel-hours |
|---|---:|---:|
| 2025-06-02 | 1 | 3 |
| 2025-06-03 | 4 | 12 |
| 2025-06-04 | 0 | 0 |
| **2025-06-05** | **18** | **64** |
| 2025-06-06 | 1 | 3 |
| **Total** | **24 of 18,260** | 82 |

- **All 24 flagged cell-hours lie in the Bornholm box** (13.5–17°E, 54.5–56.8°N); none in the
  rest of the AOI on any day.
- **Peak: 2025-06-05, 10:00–18:00 UTC.** Up to **16 affected vessels** per hour in the box,
  **9.7 %** of all vessels present; in the worst cell 6 of 16 vessels (37.5 %) jumped in one
  hour (15:00 UTC). Smaller episodes on 06-02 afternoon and 06-03 evening.
- **Convergent jumps: 0.** Jump destinations of different vessels never coincided → the
  pattern is consistent with **jamming / degraded reception, not coordinated spoofing** to a
  common false position.
- Supporting signals during the peak: the two most affected vessels reported SOG > 60 kn in
  16 % of their messages that day; a cargo ship and a tanker show 420–720° of apparent
  turning (erratic COG) inside the box at the peak.
- **MMSI-clash check:** both most-affected MMSIs carry a single name/IMO/call sign and jumped
  only while transiting the box on 06-05 (none on 06-06) → interference victims, not clashes.
  This resolves the A2 lead: the tanker that later went dark was affected by, not the cause of,
  the jump cluster.
- Land test: 541,658 normal fixes fall > 1 km inside the 1:10M land polygons (fjords and
  sounds the map closes); requiring a preceding jump reduces this to 17,093, which never drove
  a flag.
- Circles: 788 events outside ports; 138 by vessel types that should not circle, mostly
  ferries, HSC and survey vessels; 3 of them inside the box during the 06-05 peak.
- **A3 coastline re-check:** 2,316 of 2,534 encounters lie < 1 km from land; **9 of the 15
  cargo/tanker encounters** are next to the coast (berths in small harbours) → 6 remain
  offshore, including the 5 Skagen bunkering operations.
- No public ground-truth list of jamming incidents exists for validation; these are
  indicators, not confirmed events.

## A5 – ETA prediction

Cargo and tanker vessels (1,466). Arrival = WPI port-zone entry followed by a ≥ 1 h mostly
stopped stay. Samples every 30 min at sea, up to 48 h before arrival. **Time split by
arrival:** earliest 70 % of arrivals train, arrivals from 2025-06-05 04:27 UTC test.
Model: `HistGradientBoostingRegressor` on log(1 + hours), absolute-error loss; target
transform chosen on a validation split inside the training period (test never used for
choices). Output: `outputs/tables/a5_summary.json`, `a5_eta_mae_by_horizon.csv`,
`a5_feature_importance.csv`, figure `a5_eta_evaluation.png`.

- Data: 583 port stays, **414 observed arrivals**, 407 voyages with samples;
  7,871 train / **4,628 test** samples.

| MAE [h] on test | all | 0–3 h | 3–6 h | 6–12 h | 12–24 h | 24–48 h |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (great-circle distance / 1-h mean SOG) | 9.64 | **0.93** | 2.27 | 3.82 | 7.98 | 26.58 |
| **Gradient boosting** | **6.43** | 1.10 | **1.77** | **2.63** | **5.50** | **16.98** |

- **Model vs baseline: −33 % MAE overall**; better at every horizon except the last 3 h,
  where dead reckoning is already near-exact.
- **What is being compared.** Two different ETAs exist on board. The bridge ECDIS/GNSS
  computes a navigational ETA continuously (route distance-to-go / SOG); AIS does **not**
  transmit it. The AIS **ETA field** (msg 5 voyage data) is entered **manually** by the
  officer of the watch with the destination and updated "as necessary" — in practice the
  commercial ETA given to agent and port. Our baseline is the dead-reckoning analogue of the
  ECDIS ETA (with great-circle instead of route distance); "crew ETA" below is the manual
  AIS field.
- **AIS-reported (manual) ETA** (destination matched to the arrival port by UN/LOCODE or
  name, 3,084 test samples): **21.9 % were already in the past** when read (not updated).
  - All matched: AIS ETA MAE **27.15 h** (median 1.85 h) vs model 5.78 h, baseline 6.87 h.
  - Plausible only (0–14 days ahead, 2,395 samples): AIS ETA MAE **3.20 h** (median
    1.03 h) vs model 4.28 h, baseline 5.53 h.
  - → A maintained AIS ETA is the best single predictor: it carries information invisible in
    kinematics (pilot booking, berth window, planned waiting). Because it is manual, a stale
    one is the worst. A production system should combine them (AIS ETA when plausible,
    model otherwise).
- Permutation importance (log-hours MAE increase): `baseline_h` 0.37, `off_course_deg`
  0.10, `sog_kn` 0.07, `dist_nm` 0.07; vessel type ≈ 0. The model mainly corrects the
  physics baseline for heading and speed changes near port.
- Largest residual error source: vessels **waiting at anchor outside the port** for 1–2 days
  — close in distance, far in time. Not predictable from AIS kinematics alone.

## B1 – Trajectory and sensor simulation

Scenario: 1 h coastal passage, 12 kn, two 0.5°/s turns, slow-down to 8 kn, 0.4 kn current
(`b1_scenario.png`). IMU 10 Hz (truth integrated at 100 Hz, IMU outputs interval means),
GNSS 1 Hz (1.5 m white + 1.5 m Gauss-Markov τ = 60 s per axis), speed log 1 Hz (0.5 % scale
factor). Output: `outputs/tables/b1_summary.json`, `b1_free_inertial_drift.csv`.

**Error-model validation (6 h static record, tactical grade, `b1_allan_deviation.png`):**

| Term | Specified | Recovered from simulated data |
|---|---:|---:|
| Gyro ARW (ADEV at τ = 1 s) | 0.15 °/√h | **0.151 °/√h** |
| Accelerometer VRW | 0.03 m/s/√h | **0.0303 m/s/√h** |
| Allan curve vs closed-form ARW + Gauss-Markov theory | — | within 1 % for τ < 100 s, ≤ 8 % to 1000 s (unit-tested at 15 %) |

**Free-inertial drift** (INS alone from a perfect initial state, horizontal error, 200
Monte Carlo runs per grade, `b1_free_inertial_drift.png`):

| Time without GNSS | Tactical MEMS median / 95 % | Navigation FOG median / 95 % |
|---|---:|---:|
| 1 min | 5.5 m / 12.4 m | 0.5 m / 1.0 m |
| 5 min | 143 m / 309 m | 13 m / 24 m |
| 10 min | 575 m / 1.2 km | 53 m / 99 m |
| 30 min | 4.2 km / 9.3 km | 397 m / 738 m |
| 60 min | 11.1 km / 25.2 km | 1.06 km / 1.96 km |

- Sanity check: tactical 5-min median 143 m vs ½·b·t² = 132 m for a 0.3 mg bias;
  navigation grade ~1 km/h vs the classic "~1 NM/h" marine INS figure.
- A tactical MEMS IMU exceeds the GNSS error (~3 m horizontal) after **under 1 minute**;
  a navigation-grade IMU after ~3 minutes. Detecting bad GNSS quickly matters more than the
  IMU grade over short attacks.
- Without a Schuler loop (flat Earth) the > 20-min figures overstate real drift; see
  LIMITATIONS.

## B2 – Loosely coupled INS/GNSS EKF

Error-state EKF, 8 states (δp 2, δv 2, δψ, accelerometer bias 2, gyro bias); INS at 10 Hz,
covariance and GNSS position update at 1 Hz, Joseph form. Tactical MEMS IMU, 1 h scenario,
**30 Monte Carlo runs per experiment**, statistics after the first 10 min.
Output: `outputs/tables/b2_summary.json`, figures `b2_single_run.png`, `b2_consistency.png`.
Linearisation verified by a unit test: nonlinear error propagation matches Φ·δx.

| Experiment | Position RMS | Raw GNSS RMS | Heading RMS | Accel bias RMS | Gyro bias RMS | Avg NEES (8) | Avg NIS (2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| White GNSS noise | **0.59 m** | 2.99 m | 0.53° | 0.016 mg | 1.96 °/h | **6.8** (0 % epochs above band) | **1.98** (95 % in band) |
| Realistic GNSS (white + GM τ = 60 s) | 2.31 m | 2.99 m | 0.78° | 0.027 mg | 2.42 °/h | **74.7** (100 % above) | 1.42 |
| Realistic, R inflated ×1.5 σ | 2.29 m | 2.99 m | 0.70° | 0.024 mg | 2.20 °/h | 39.5 (100 % above) | 0.66 |

- With white GNSS noise the filter is **consistent** (slightly conservative: the GM biases
  are modelled as random walks) and cuts position error **5×** relative to raw GNSS.
  Accelerometer biases are estimated to ~0.02 mg against a 0.3 mg turn-on spread.
- With **time-correlated GNSS error** the filter follows the slow error instead of averaging
  it: accuracy gain drops to 1.3× and it becomes **~10× overconfident** (NEES 75 vs 8)
  while the **NIS looks normal or even low** (1.4). Innovation monitoring cannot see a slow,
  smooth error — the same blind spot a slow-drift (meaconing) spoofer exploits.
- Inflating R, the usual engineering fix, only halves the NEES; the principled fix is
  state augmentation with GNSS-error states (not done; see LIMITATIONS).
- **Observability:** heading σ drops sharply in each turn and grows on straight legs; the
  gyro bias is learned in steps, only during manoeuvres (`b2_single_run.png`).

## B3 – GNSS attacks on the unprotected EKF

Attacks on the GNSS position stream, 40–55 min (15 min, final straight leg, filter converged),
offset towards east. **20 Monte Carlo runs per case, same seeds across cases** (paired).
Tactical MEMS IMU, realistic GNSS. Per-epoch test: NIS > χ²₂(99 %) = 9.21.
Output: `outputs/tables/b3_summary.json`, `b3_drift_stealth.csv`, figures
`b3_attacks_unprotected.png`, `b3_drift_stealth.png`.

| Case | Max error during attack (median / 95 %) | Error at attack end | Mean error 5 min after | Attack epochs with NIS > 9.21 |
|---|---:|---:|---:|---:|
| No attack | 2.1 m mean, 4.1 m p95 | — | — | 0.14 % (all epochs) |
| **Jump 180 m** | 233 m / 236 m | 180 m | 28.9 m | 32.4 % |
| **Slow drift 0.2 m/s** (→ 180 m) | 180 m / 182 m | 180 m | 28.6 m | **0.4 %** |
| **Jamming 15 min** | 187 m / 289 m | 187 m | **1.9 m** | n/a (no fixes) |

- **Jump:** NIS spikes to ~10⁴ at the step — trivially detectable — but the unprotected filter
  accepts it, is captured within 2–3 min (NIS returns to normal while 180 m off) and
  overshoots to 233 m because velocity and biases are corrupted.
- **Slow drift (0.2 m/s ≈ 0.4 kn):** the filter follows the full 180 m while the per-epoch
  χ² test fires on 0.4 % of attack epochs, indistinguishable from no attack (0.14 %).
- **After either spoofing attack ends**, the return to true GNSS is itself a step: the filter
  rings for ~5 min (29 m mean error) because velocity and bias states were corrupted.
- **Jamming** is the least dangerous: the filter knows it has no GNSS and coasts on an INS
  calibrated before the outage — **187 m after 15 min vs 1,487 m** for the same IMU
  uncalibrated (B1 free inertial), **8× better** — and re-converges immediately.

**Drift stealth sweep** (10 runs per rate, `b3_drift_stealth.png`):

| Drift rate | Offset after 15 min | Filter error at end | Attack epochs with NIS > 9.21 |
|---|---:|---:|---:|
| 0.05 m/s | 45 m | 45 m | 0.1 % |
| 0.1 m/s | 90 m | 90 m | 0.1 % |
| **0.2 m/s** | **180 m** | **180 m** | **0.3 %** |
| 0.3 m/s | 270 m | 270 m | 0.9 % |
| 0.5 m/s | 450 m | 450 m | 4.8 % |
| 1.0 m/s | 900 m | 899 m | 18.1 % |

→ The filter is dragged along the full spoofed offset at every rate; below ~0.3 m/s the
per-epoch innovation test is effectively blind. This motivates the windowed test in B4.

## B4 – Integrity monitoring: detect, reject, coast, recover

Monitor = per-epoch χ² gate (NIS > 13.82, 3 consecutive → declare) + **backup-INS window
test** (a snapshot every 30 s runs on the IMU only; at age 120 s the GNSS fix is tested
against it, T = r′S⁻¹r) with **rollback** onto the clean backup on detection. While GNSS is
distrusted: coasting with **speed-log + non-holonomic aiding** (current estimated while GNSS
was trusted), 5-min quarantine, re-admission after 10 consecutive consistent fixes within
60 m, 30-min timeout with operator alert. Output: `outputs/tables/b4_summary.json`, figures
`b4_protected_vs_unprotected.png`, `b4_window_test_calibration.png`.

**Calibration** (20 attack-free runs, independent seeds, 2,000 window tests): mean T = 6.94
vs 2 for white-noise theory → fitted scale α = 3.47 → threshold **63.9** for p_fa = 1e-4 per
test; 1 of 2,000 calibration tests (0.05 %) exceeded it.

**Evaluation** (10 runs per case, same seeds with and without the monitor):

| Case | Unprotected: error at end / 5 min after | Protected: error at end / 5 min after | Detection |
|---|---:|---:|---|
| No attack | 2.1 m mean | 2.1 m mean (unchanged) | **0 false alarms in 10 h** |
| **Jump 180 m** | 180 m / 29 m | **34 m / 3.1 m** | **10/10, 1 s** |
| **Drift 0.2 m/s**, detected runs | 180 m / 29 m | **36 m / 2.8 m** | **6/10**, median 121 s (max 211 s) |
| Drift 0.2 m/s, missed runs | 180 m / 29 m | 181 m / **209 m** | 4/10 missed during attack |
| Jamming 15 min | 187 m / 1.9 m | 187 m / 1.9 m | not needed (no fixes) |

- **Jumps** are neutralised: detected in the first second, the filter coasts on INS + speed
  log at ~10–35 m instead of being captured at 180–233 m, and re-admits GNSS within minutes
  of the attack ending.
- **Slow drift** at 0.2 m/s sits at the detection limit of a tactical MEMS IMU. When caught,
  rollback removes the contamination (36 m vs 180 m). When missed, the end of the attack
  looks like a jump of the genuine signal, which is then rejected: protection makes the
  aftermath worse (209 m vs 29 m). This is the fundamental "who is lying, GNSS or INS?"
  ambiguity; it needs an independent reference or the operator (see LIMITATIONS).
- Development findings kept as tests: re-admission without a quarantine let the drifting
  spoofer straight back in; a 10-min timeout re-admitted a 15-min jump spoofer.

## B5 – Monte Carlo evaluation of the integrity monitor

Parallel Monte Carlo (4 worker processes; parallel and serial results verified identical):
**210 evaluation runs + 40 calibration runs** of the 1-h scenario. Each run executes the
unprotected and protected filters on identical sensors and initial errors (paired).
Window threshold calibrated per IMU grade on independent attack-free runs, scale matched at
the 99th percentile, p_fa = 1e-4 per test. Monitor armed after 10 min. 10 runs per sweep
point (95 % Wilson intervals in the figure). Output: `outputs/tables/b5_summary.json`,
`b5_sweeps.csv`, figures `b5_detection_curves.png`, `b5_error_reduction.png`.

| | Tactical MEMS | Navigation FOG |
|---|---:|---:|
| Calibrated window threshold (scale α) | 62.9 (3.42) | 44.2 (2.40) |
| **False alarms, 20 h attack-free** | **0** | **1 (0.05 /h)** |
| Mean error attack-free, protected vs unprotected | 2.07 vs 2.07 m | 1.91 vs 1.85 m |

**Slow drift (15-min attack): detection probability / median delay / median error at end
(unprotected → protected)**

| Drift rate | Tactical MEMS | Navigation FOG |
|---|---|---|
| 0.05 m/s | 0 % — 45 → 45 m | 10 % — 45 → 45 m |
| 0.1 m/s | 0 % — 90 → 90 m | 60 %, 120 s — 90 → 89 m |
| 0.2 m/s | 70 %, 120 s — 180 → 106 m | **100 %, 70 s — 180 → 11 m** |
| 0.3 m/s | **100 %, 90 s — 270 → 18 m** | 100 %, 39 s — 269 → 12 m |
| 0.5 m/s | 100 %, 25 s — 450 → 29 m | 100 %, 23 s — 448 → 12 m |
| 1.0 m/s | 100 %, 13 s — 899 → 19 m | 100 %, 12 s — 897 → 13 m |

**Jumps (tactical MEMS):** 5 m: 10 % detected (inside GNSS noise); **10–180 m: 100 % within
2 s**. Error at end of attack: 180 m → **24 m**; for 10–50 m jumps no gain (protected ≈
offset), because the 15-min coasting error is of the same size as the jump.

**Key findings**
- **IMU grade sets the detection limit:** reliable (100 %) drift detection from **0.3 m/s
  with a tactical MEMS** vs **0.2 m/s with a navigation-grade FOG**, and the FOG detects
  ~1.5–2× faster at the same rate (70 vs 120 s at 0.2 m/s; 39 vs 90 s at 0.3 m/s).
- Above the limit the monitor turns a spoofed offset of 270–900 m into **12–29 m** at the end
  of a 15-min attack, and the filter is back at ~2–3 m within 5 min of the attack ending.
- **Grey zone:** spoofed offsets comparable to what the INS + current uncertainty would
  drift on its own (≲ 50 m in 15 min) are either undetectable or indistinguishable after the
  quarantine; protection does not pay off there.

## B – INS/GNSS resilience (Monte Carlo)
_superseded by B5 above_
