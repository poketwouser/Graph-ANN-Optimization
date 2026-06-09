# AdaptiveVamana — Benchmark Report

**Dataset**: SIFT1M (1,000,000 × 128 base, 10,000 queries, ground-truth top-100).
**Hardware**: Windows 11, 16 logical cores. **Compiler**: MSYS2 g++ 15.2.0, `-O3 -march=native`.
**Index build**: R=32, L=75, α=1.2, γ=1.5 (avg out-degree ≈ 33).
**Metric definitions**: Recall@k = |result top-k ∩ GT top-k| / k, averaged over 10k queries.
AvgCmps = mean L2 distance computations/query (incl. centroid comparisons for routed modes).
Latency = **single-thread** mean per query (low variance); QPS measured separately.
All rows below use the same machine/harness, so cross-rows are directly comparable.

> No numbers are fabricated. Every row was produced by `search_index` / `build_*` / `refine_index`.

---

## 1. Baseline (original code) — operating curve

Legacy `std::set` search engine, original graph.

| L | Recall@10 | Recall@100 | AvgCmps | AvgLat µs | P95 µs | P99 µs | QPS |
|---|-----------|------------|---------|-----------|--------|--------|-----|
| 10 | 0.7750 | 0.0977 | 642.5 | 275.4 | 415.3 | 554.5 | 3617 |
| 20 | 0.8898 | 0.1992 | 883.4 | 391.9 | 569.4 | 728.6 | 2541 |
| 30 | 0.9315 | 0.2994 | 1100.5 | 501.9 | 706.5 | 921.1 | 1984 |
| 50 | 0.9661 | 0.4997 | 1510.4 | 707.7 | 972.3 | 1234.8 | 1408 |
| 75 | 0.9820 | 0.7486 | 1987.7 | 950.4 | 1286.5 | 1624.0 | 1048 |
| 100 | 0.9890 | 0.9549 | 2437.2 | 1194.3 | 1601.6 | 2028.1 | 834 |
| 150 | 0.9939 | 0.9767 | 3271.9 | 1679.3 | 2248.7 | 2707.3 | 594 |
| 200 | 0.9960 | 0.9858 | 4046.1 | 2188.7 | 2897.2 | 3434.5 | 456 |

**Exhaustive (brute-force) reference**: 1,000,000 distance comps/query; ~362,000 µs single-thread
(numpy), ~23,000 µs on 16-core BLAS.

---

## 2. F0 — Optimized search engine (fast vs legacy, identical recall)

Same index, same algorithm, different data structures. Recall & AvgCmps are **bit-identical**;
only latency differs. Speedup grows with L (legacy's O(L²) frontier scan + per-query 1 MB
`visited` allocation).

| L | Recall@10 | AvgCmps | Legacy µs | Fast µs | Speedup |
|---|-----------|---------|-----------|---------|---------|
| 50 | 0.9661 | 1510.4 | 705.7 | 666.1 | 5.6% |
| 75 | 0.9820 | 1987.7 | 950.4 | 856.7 | 9.9% |
| 100 | 0.9890 | 2437.2 | 1194.3 | 1055.4 | 11.6% |
| 150 | 0.9939 | 3271.9 | 1679.3 | 1431.6 | 14.8% |
| 200 | 0.9960 | 4046.1 | 2188.7 | 1799.5 | 17.8% |

---

## 3. F3 — Adaptive beam width

Per-query convergence-based early stop. Effective beam (nodes expanded) is bell-shaped, e.g.
config 40/100/12/0.003: min 40, p50 68, p90 89, max 145.

| min/max/pat/ε | R@10 | AvgCmps | Lat µs | Fixed-L @ same R@10 | Cmps saved |
|---|---|---|---|---|---|
| 50/100/10/0.004 | 0.9705 | 1564.7 | 704 | L≈55 → ~1640 | ~5% |
| 40/100/12/0.003 | 0.9770 | 1716.3 | 757 | L≈67 → ~1837 | ~6.6% |
| 40/160/15/0.002 | 0.9842 | 1980.8 | 896 | L≈80 → ~2100 | ~6% |

Lesson: tighter `max_beam` lowers latency (pool insertion is O(pool)) — 50/100 (704 µs) beats
50/160 (725 µs) at identical recall/cmps.

---

## 4. F1 — K-Means query routing

### 4a. IVF-restrict sweep (L=50). Baseline = 0.9661 / 1510 / 708 µs.

| K \ nprobe | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 16 | 0.7423 / 1018 | 0.8990 / 1184 | 0.9602 / 1305 | **0.9664 / 1341** |
| 32 | 0.6727 / 916 | 0.8469 / 1078 | 0.9402 / 1230 | 0.9646 / 1326 |
| 64 | 0.6114 / 831 | 0.7875 / 985 | 0.9090 / 1149 | 0.9565 / 1284 |
| 128 | 0.5440 / 780 | 0.7243 / 922 | 0.8667 / 1085 | 0.9405 / 1246 |
| 256 | 0.4793 / 789 | 0.6553 / 916 | 0.8077 / 1073 | 0.9073 / 1242 |

(cells = Recall@10 / AvgCmps.) Best iso-recall: **K=16, nprobe=8** → 0.9664 @ 1341 (−11% cmps).
Higher K at low nprobe = recall ceiling (IVF effect).

### 4b. Route-only (full graph, medoid seeds) — preserves recall at every L

| L | Baseline R@10/cmps/µs | RouteK64np2 R@10/cmps/µs | Δcmps / Δlat |
|---|---|---|---|
| 30 | 0.9315 / 1100 / 461 | 0.9338 / 949 / 377 | −14% / −18% |
| 50 | 0.9661 / 1510 / 666 | 0.9664 / 1356 / 554 | −10% / −17% |
| 75 | 0.9820 / 1988 / 857 | 0.9820 / 1831 / 748 | −8% / −13% |
| 100 | 0.9890 / 2437 / 1055 | 0.9890 / 2279 / 947 | −6% / −10% |

---

## 5. F2 — Multi-entry (hubs + random on top of medoid). Route-only K64/np2, L=50.

| hubs / random | R@10 | AvgCmps |
|---|---|---|
| 0/0 (medoid only) | 0.9664 | 1355.8 |
| 3/0 | 0.9664 | 1358.4 |
| 0/3 | 0.9664 | 1358.2 |
| 3/2 | 0.9664 | 1360.2 |
| 5/5 | 0.9664 | 1364.5 |

Null result — medoid routing already lands the search next to the answer.

---

## 6. F5 / F6 — Graph quality (plain fast search)

| Variant | avg deg | R@10 L=30 | L=50 | L=75 |
|---|---|---|---|---|
| baseline | 33.04 | 0.9315 | 0.9661 | 0.9820 |
| F5 diversity 0.1 | 33.39 | 0.9326 | 0.9668 | 0.9819 |
| F5 diversity 0.3 | 33.63 | 0.9324 | 0.9663 | 0.9819 |
| F6 noise 0.05 (+2/node) | 35.10 | **0.9342** | 0.9667 | 0.9820 |
| F6 noise 0.10 (+3/node) | 36.09 | 0.9338 | 0.9668 | 0.9819 |

F5 within build noise (~±0.002). F6 0.05 = small low-L win; 0.10 inflates high-L cmps.

---

## 7. F4 — Edge-usage refinement (held-out learn set → eval queries)

| config | avg deg | R@10 L=50 | cmps L=50 | lat µs | graph MB |
|---|---|---|---|---|---|
| baseline | 33.0 | 0.9661 | 1510 | 666 | 149 |
| keep_min=8 (aggressive) | 11.6 | 0.8469 | 919 | 465 | 67 |
| keep_min=26 | 27.4 | 0.9661 | 1376 | 590 | 127 |
| **keep_min=20 (chosen)** | **24.4** | **0.9661** | **1322** | **587** | **116** |

deg-24: recall preserved at every L, −12% cmps, −12% latency, −22% memory. Aggressive pruning
collapses recall (tail-query edges matter).

---

## 8. FINAL — AdaptiveVamana (F0+F4+F1+F3 combined)

Refined deg-24 graph + route-only K=64/nprobe=2 + adaptive beam.

| System | config | R@10 | R@100 | AvgCmps | AvgLat µs | P95 | P99 | QPS |
|--------|--------|------|-------|---------|-----------|-----|-----|-----|
| Baseline (legacy, L=50) | — | 0.9661 | 0.4997 | 1510.4 | 707.7 | 972 | 1235 | 1408 |
| **AV-Fast** | 35/90/10/0.004 | 0.9658 | 0.8522 | **1157.7** | **495.2** | 761 | 956 | 2002 |
| **AV-Balanced** | 40/100/12/0.003 | 0.9759 | 0.9036 | 1344.2 | 609.8 | 940 | 1246 | 1627 |
| **AV-Accurate** | 70/170/14/0.002 | **0.9838** | 0.9345 | 1631.5 | 719.4 | 1072 | 1333 | 1380 |

### Final Pareto curve (sweep of the combined system)

| min/max/pat/ε (np=2) | R@10 | AvgCmps | AvgLat µs |
|---|---|---|---|
| 25/70/8/0.006 | 0.9434 | 921.4 | 387.9 |
| 30/80/10/0.004 | 0.9631 | 1117.6 | 472.3 |
| 35/90/10/0.004 | 0.9658 | 1157.7 | 495.2 |
| 40/100/12/0.003 | 0.9759 | 1344.2 | 609.8 |
| 60/130/12/0.003 | 0.9784 | 1441.0 | 639.2 |
| 70/170/14/0.002 | 0.9838 | 1631.5 | 719.4 |
| 90/200/16/0.0015 | 0.9877 | 1895.9 | 836.6 |

### Headline deltas

| Comparison | Recall@10 | AvgCmps | Latency |
|---|---|---|---|
| AV-Fast vs baseline (iso-recall) | 0.9661→0.9658 | **−23.4%** | **−30.0%** |
| AV-Balanced vs baseline | 0.9661→**0.9759** | −11.0% | −13.8% |
| AV-Accurate vs baseline | 0.9661→**0.9838** | +8.0%* | +1.7%* |
| AV-Fast vs exhaustive | — | **864× fewer** | **~730× faster** |

\* AV-Accurate is compared at *higher recall*; against the baseline curve **at 0.984 recall**
(~L=78) it is ~−20% cmps and ~−20% latency.
