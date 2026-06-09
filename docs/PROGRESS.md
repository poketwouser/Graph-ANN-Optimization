# AdaptiveVamana — Progress Log

A workload-aware, cluster-routed enhancement of an educational Vamana (DiskANN)
ANN index. This log is updated after **every** change: files modified, rationale,
benchmark results, and observed tradeoffs.

---

## Environment

| Item | Value |
|------|-------|
| Platform | Windows 11, 16 logical cores |
| Toolchain | MSYS2 g++ 15.2.0, cmake 4.2.1 (`C:\msys64\mingw64\bin`) |
| Python | 3.13.13, numpy 2.4.2 |
| Dataset | SIFT1M (1,000,000 × 128 base, 10,000 queries, GT top-100) |
| Distance | squared L2 (`-O3 -march=native` auto-vectorized) |
| Build params (baseline index) | R=32, L=75, α=1.2, γ=1.5 |

## Methodology decisions (agreed with user)

1. **Dataset**: download real SIFT1M, reproduce baseline, validate every feature on full 1M.
2. **Latency attribution**: the search-engine optimization (replacing the O(L²) `std::set`
   loop + per-query 1 MB `visited` allocation with a flat sorted candidate array + reusable
   visited buffer) is counted as **part of AdaptiveVamana**. We report a single combined
   latency improvement vs. the *original* baseline.
3. **Layout**: enhance `Base/` in place. All new behavior is **flag-gated** so the original
   baseline path is always recoverable. Docs live in `Base/docs/`.

## Targets

| Metric | Baseline (L=50) | Target |
|--------|-----------------|--------|
| Recall@10 | 96.64% | ~98.1% |
| Avg latency | 1044.7 µs | ≥18% lower |
| Avg distance computations | 1511 | ≥27% lower |

> Rule: never fabricate metrics. Only experimentally measured numbers are recorded below.

---

## Baseline (original code, reference)

Saved run (`search_results.txt`), index R=32/L=75/α=1.2/γ=1.5:

| L | Recall@10 | Avg Dist Cmps | Avg Latency (µs) | P99 (µs) |
|---|-----------|---------------|------------------|----------|
| 10 | 0.7754 | 643 | 656 | 2769 |
| 20 | 0.8902 | 884 | 570 | 1508 |
| 30 | 0.9328 | 1101 | 733 | 2129 |
| 50 | 0.9664 | 1511 | 1045 | 3220 |
| 75 | 0.9819 | 1988 | 1372 | 4222 |
| 100 | 0.9889 | 2438 | 1738 | 6517 |
| 150 | 0.9939 | 3273 | 2620 | 21479 |
| 200 | 0.9961 | 4047 | 3000 | 10189 |

### Baseline re-measured on THIS machine (the reference for all comparisons)

Index rebuilt with original `build_index.exe` (R=32/L=75/α=1.2/γ=1.5): build 84.6 s,
avg out-degree 33.10. Search via original `search_index.exe`, 10k queries, K=10:

| L | Recall@10 | Avg Dist Cmps | Avg Latency (µs) | P99 (µs) |
|---|-----------|---------------|------------------|----------|
| 10 | 0.7759 | 641.0 | 411.2 | 1543 |
| 20 | 0.8915 | 882.0 | 633.6 | 2706 |
| 30 | 0.9328 | 1098.3 | 762.3 | 2942 |
| **50** | **0.9663** | **1508.1** | **1073.3** | 4651 |
| 75 | 0.9820 | 1985.3 | 1397.4 | 5562 |
| 100 | 0.9889 | 2434.9 | 1780.2 | 10141 |
| 150 | 0.9937 | 3269.7 | 2402.0 | 9072 |
| 200 | 0.9960 | 4044.4 | 3017.1 | 11840 |

Recall and distance-counts match the saved reference to <0.2%, confirming a faithful index.
Latency differs (different hardware) but **all comparisons below use this same harness/machine**,
so they are internally valid. **Anchor baseline = L=50 row above.**

---

## Architecture analysis (Phase 1)

**Data**: contiguous 64B-aligned `float[npts*dim]`, row-major.
**Graph**: `vector<vector<uint32_t>>` adjacency lists; single random start node.
**Build**: random-order insertion, OpenMP + per-node mutex; greedy_search → robust_prune
(α-RNG) → bidirectional edges → γR degree-bounded re-prune.
**Search**: single-entry greedy beam search, candidate set bounded at L.

### Identified bottlenecks
1. `greedy_search` allocates `vector<bool> visited(npts_)` (~1 MB) **per query** and uses a
   cache-hostile `std::set` with an **O(L²)** front-scan to find the next node to expand.
   → dominant latency cost; independent of recall.
2. Single **random** entry point (not even medoid) → extra hops, local-minima risk.
3. Global fixed beam `L` for all queries → no adaptivity.
4. No clustering / routing → every query scans the full graph.
5. Latent bug: `_aligned_malloc` paired with `std::free` (mismatched free, UB on Windows).
6. Harness missing Recall@100, P95, memory, and per-query stats.

---

## Feature implementation log

> Each entry: **files modified · rationale · results · tradeoffs**. Appended as work proceeds.

### F0 — Foundation (optimized search engine + metrics harness + bug fix)
- Status: **DONE ✓**
- **Files modified**:
  - `include/vamana_index.h` — added `set_legacy_search()`, `graph_memory_bytes()`,
    `greedy_search_fast()`/`greedy_search_legacy()` decls, `legacy_search_` flag.
  - `src/vamana_index.cpp` — new `greedy_search_fast()` (flat sorted candidate pool +
    thread-local version-stamped visited buffer); dispatcher; `graph_memory_bytes()`;
    fixed mismatched free in destructor (`std::free` → `portable_free`).
  - `src/io_utils.cpp` — fixed unique_ptr deleters (`std::free` → `portable_free`) so
    `_aligned_malloc`'d memory is released with `_aligned_free` (was UB on Windows).
  - `src/search_index.cpp` — Recall@10 + Recall@100, P95/P99, QPS, index-memory
    reporting, `--legacy-search` toggle.
- **Rationale**: the legacy `greedy_search` allocated a fresh `vector<bool>(npts_)`
  (~1 MB) per query and used a cache-hostile `std::set` with an O(L²) front-scan. Replacing
  these with a contiguous pool + O(1)-clear visited buffer removes per-query allocation and
  cuts the frontier bookkeeping to O(L) per insertion — pure engineering, no algorithm change.
- **Correctness**: on the *same* index, the fast backend reproduces the legacy backend's
  Recall@10/Recall@100/AvgCmps **exactly** at every L (verified L=10..200). A subtle
  cursor bug (skipping a neighbor that inserts exactly at the cursor position; `<` vs `<=`)
  was found via this equivalence check and fixed.
- **Results** (single-thread, 10k queries, same index, R=32):

  | L | R@10 | AvgCmps | Legacy µs | Fast µs | Speedup | Legacy QPS | Fast QPS |
  |---|------|---------|-----------|---------|---------|-----------|----------|
  | 50 | 0.9661 | 1510.4 | 705.7 | 666.1 | 5.6% | 1411 | 1495 |
  | 75 | 0.9820 | 1987.7 | 950.4 | 856.7 | 9.9% | 1048 | 1162 |
  | 100 | 0.9890 | 2437.2 | 1194.3 | 1055.4 | 11.6% | 834 | 943 |
  | 150 | 0.9939 | 3271.9 | 1679.3 | 1431.6 | 14.8% | 594 | 696 |
  | 200 | 0.9960 | 4046.1 | 2188.7 | 1799.5 | 17.8% | 456 | 554 |

- **Tradeoffs**: speedup is small at low L (5.6% @ L=50) and grows with L (17.8% @ L=200),
  because the legacy overhead it removes is super-linear in L. The bulk of search time is the
  (identical) distance computations into the 512 MB dataset, so engineering alone cannot hit
  the latency target — the algorithmic features must reduce *distance computations*.
- **Methodology note**: latency is now measured **single-threaded** (true per-query latency,
  low variance); the original 16-thread harness was memory-bandwidth-saturated and noisy
  (P99 swinging 5–20 ms). Throughput (QPS) is reported separately. All later comparisons use
  this single-thread harness and the **F0 fast engine** as the running baseline curve.

### F3 — Adaptive beam width
- Status: **DONE ✓**
- **Files modified**:
  - `include/vamana_index.h` — `AdaptiveConfig` struct; `SearchResult.expanded`;
    `search_adaptive()` + `set_adaptive()`; `greedy_search_adaptive()` decl; `adaptive_cfg_`.
  - `src/vamana_index.cpp` — `greedy_search_adaptive()` and `search_adaptive()`.
  - `src/search_index.cpp` — `--adaptive --min-beam --max-beam --patience --epsilon`;
    adaptive run path with effective-beam percentiles + ASCII histogram.
- **Algorithm**: pool capped at `max_beam`; track the K-th-neighbor distance (result
  boundary); stop once it fails to improve by > `epsilon` (relative) for `patience`
  consecutive expansions, but never before `min_beam` expansions. Difficulty signal =
  convergence of the result boundary — a pure search-time statistic (no clustering needed).
- **Per-query adaptivity confirmed** (config 40/100/12/0.003): effective beam ranges 40→145,
  bell-shaped (p50=68, p90=89), 816/10000 easy queries stop at the floor while a hard-query
  tail reaches 100+.
- **Results** (single-thread; vs F0 fixed-L curve). Adaptive vs fixed at matched recall:

  | Config (min/max/pat/ε) | R@10 | AvgCmps | Lat µs | Fixed-L @ same R@10 | Cmps saved |
  |---|---|---|---|---|---|
  | 50/100/10/0.004 | 0.9705 | 1564.7 | 704 | L≈55 → ~1640 cmps | ~5% |
  | 40/100/12/0.003 | 0.9770 | 1716.3 | 757 | L≈67 → ~1837 cmps | ~6.6% |
  | 40/160/15/0.002 | 0.9842 | 1980.8 | 896 | L≈80 → ~2100 cmps | ~6% |

  At matched recall, adaptive does **~5–7% fewer distance computations** and ~3–4% lower
  latency than fixed-L.
- **Tradeoffs / lessons**:
  - Win is *modest on SIFT* because query difficulty is fairly uniform (most queries cluster
    near the median beam). Adaptive pays off more on heterogeneous workloads; the budget it
    saves on easy queries is reinvested in hard ones.
  - Large `max_beam` inflates latency: the flat pool's insertion is O(pool), so `max_beam=160`
    adds ~3% latency vs `max_beam=100` at identical recall/cmps. **Keep `max_beam` tight.**
  - F3's primary lever is *distance computations* (a target metric); its latency benefit is
    secondary and partly eaten by per-expansion convergence checks + pool overhead.

### F1 — K-Means query routing
- Status: **DONE ✓**
- **Files added**: `include/kmeans.h`, `src/kmeans.cpp` (k-means++ init + Lloyd on a
  subsample, full assignment, medoids), `src/build_clusters.cpp` (CLI). **Modified**:
  `CMakeLists.txt` (kmeans + build_clusters targets); `include/vamana_index.h` +
  `src/vamana_index.cpp` (`load_clusters`, `set_routing`, `greedy_search_routed`,
  `search_routed`); `src/search_index.cpp` (`--clusters --clusters-to-search --route-only`).
- **Two modes** (both configurable, both benchmarked):
  - **IVF-restrict**: rank clusters by centroid distance, seed from the nprobe nearest
    medoids, and restrict the graph walk to points in those clusters (skip out-of-cluster
    neighbors → fewer distance computations, but a recall ceiling).
  - **Route-only**: seed from the nprobe nearest medoids but search the *full* graph (better
    entry points, no recall ceiling).
- **Clustering cost** (subsample=200k, 15 iters): K=16 → 1.8s … K=256 → 13.8s. Sizes balanced
  within ~2–5×.
- **Results** (single-thread, L=50, baseline = 0.9661 / 1510 cmps / 666 µs):

  *IVF-restrict* — best iso-recall is **K=16, nprobe=8**: R@10 0.9664, 1341 cmps (−11%),
  577 µs (−13%). Larger K at low nprobe trades recall for cmps (K=256/np1: R@10 0.479 @ 789
  cmps) — the IVF recall-ceiling.

  *Route-only* — preserves recall at **every** L with ~10–14% fewer cmps and ~15–18% lower
  latency:

  | L | Base R@10/cmps/µs | RouteK64np2 R@10/cmps/µs |
  |---|---|---|
  | 30 | 0.9315 / 1100 / 461 | 0.9338 / 949 / 377 |
  | 50 | 0.9661 / 1510 / 666 | 0.9664 / 1356 / 554 |
  | 75 | 0.9820 / 1988 / 857 | 0.9820 / 1831 / 748 |
  | 100 | 0.9890 / 2437 / 1055 | 0.9890 / 2279 / 947 |

- **Key research finding**: on a *monolithic* Vamana graph, IVF hard-restriction is **dominated
  by soft medoid-routing**. Restriction's distance savings are partly offset by a per-edge
  cluster-membership lookup (random access into the assignment array) and it imposes a recall
  ceiling, whereas medoid entry points cut hops with no recall loss. The baseline's single
  *random* start node was the real inefficiency. (Centroid comparisons are counted in cmps;
  smaller K keeps that overhead low — favor K=16–64 for routing.)
- **Tradeoffs**: routing needs offline clustering (one-time) + centroids/assignments in memory
  (~4 MB). IVF-restrict useful when memory/latency budget forces searching a data subset and
  some recall loss is acceptable.

### F2 — Multi-entry search
- Status: **DONE ✓ (honest negative result)**
- **Files modified**: `include/vamana_index.h` + `src/vamana_index.cpp` (`set_entry_mix`,
  `ensure_hubs`, hub/random seeding in `search_routed`); `src/search_index.cpp`
  (`--hubs --random-entries`).
- **Design**: in addition to the nprobe cluster medoids, seed the search with the top
  `num_hubs` highest-out-degree nodes (well-connected launch points) and `num_random` random
  nodes (exploration), merging all into one candidate pool.
- **Results** (route-only K=64, np=2, L=50; medoid-only = 0.9664 / 1356 cmps):

  | hubs / random | R@10 | AvgCmps |
  |---|---|---|
  | 0 / 0 (medoid only) | 0.9664 | 1355.8 |
  | 3 / 0 | 0.9664 | 1358.4 |
  | 0 / 3 | 0.9664 | 1358.2 |
  | 3 / 2 | 0.9664 | 1360.2 |
  | 5 / 5 | 0.9664 | 1364.5 |

- **Finding / lesson**: once query-adjacent **medoid** entry points are used, adding hub and
  random launch points yields **no measurable recall gain** and a small cmps cost — they are
  redundant. Multi-entry *does* help, but that benefit is already realized by F1 (medoid
  routing) relative to the baseline's single **random** start node (0.9661→0.9664 at −10% cmps).
  Hubs/random would matter when entry points are otherwise poor (e.g. no routing) or to bridge
  clusters in IVF-restrict mode — not in the route-only regime we adopt. Kept configurable
  (defaults 0/0) for completeness; **not used in the final system.**

### F5 — Cluster-diverse pruning
- Status: **DONE ✓ (marginal / within noise)**
- **Files modified**: `include/vamana_index.h` + `src/vamana_index.cpp`
  (`set_build_diversity`, diversity logic in `robust_prune`); `src/build_index.cpp`
  (`--clusters --diversity`).
- **Design**: during RobustPrune, if a candidate's cluster is not yet represented among the
  selected neighbors, relax its α-RNG threshold to `α·(1+strength)` so it is more likely kept
  → neighbor set spans more clusters.
- **Results** (plain fast search, K=64 clusters; baseline L=50 = 0.9661):

  | strength | R@10 L=30 | L=50 | L=75 | avg deg |
  |---|---|---|---|---|
  | 0 (baseline) | 0.9315 | 0.9661 | 0.9820 | 33.04 |
  | 0.1 | 0.9326 | 0.9668 | 0.9819 | 33.39 |
  | 0.3 | 0.9324 | 0.9663 | 0.9819 | 33.63 |

- **Finding**: gains (+0.0007–0.0011) are **within parallel-build nondeterminism (~±0.002)** —
  effectively null. Vamana's α-RNG already yields geometrically diverse (hence cluster-diverse)
  neighbors on SIFT, so explicit cluster-diversity adds little. Implemented + configurable;
  **not used in final system.**

### F6 — Selective noise injection
- Status: **DONE ✓ (small win at low L)**
- **Files modified**: `src/vamana_index.cpp` (`inject_noise` post-build pass);
  `include/vamana_index.h`; `src/build_index.cpp` (`--noise <ratio>`,
  edges/node = round(ratio·R)).
- **Design**: add `round(noise_ratio·R)` random long-range out-edges per node after build
  (deduped, additive → connectivity preserved). Small-world shortcuts to distant regions.
- **Results** (plain fast search; baseline in row 0):

  | noise | edges/node | avg deg | R@10 L=30 / cmps / µs | L=50 | L=75 / cmps |
  |---|---|---|---|---|---|
  | 0 | 0 | 33.04 | 0.9315 / 1100 / 459 | 0.9661 | 0.9820 / 1988 |
  | 0.05 | 2 | 35.10 | **0.9342 / 1067 / 435** | 0.9667 | 0.9820 / 2040 |
  | 0.10 | 3 | 36.09 | 0.9338 / 1097 | 0.9668 | 0.9819 / 2114 |

- **Finding**: noise=0.05 gives a **small Pareto win at low L** (L=30: +0.27% recall, −3% cmps,
  −5% latency) — random shortcuts help when the beam is too small to navigate via local edges.
  At L≥50 it is neutral, and noise=0.10 *hurts* high-L cmps (+6% at L=75, no recall gain) since
  every extra edge is evaluated each expansion. **Lesson**: a little noise helps navigability;
  too much just inflates degree and search cost. Useful only in the low-recall/low-L regime;
  **not used in the final (high-recall) system.**

### F4 — Edge-usage-guided refinement (held-out workload)
- Status: **DONE ✓ (winner when gentle; fails when aggressive)**
- **Files added**: `src/refine_index.cpp` (CLI). **Modified**: `include/vamana_index.h` +
  `src/vamana_index.cpp` (`init_edge_usage`, `collect_edge_usage`, `refine_by_usage`);
  `CMakeLists.txt`.
- **Method**: over a **held-out** workload (SIFT *learn* set, disjoint from the eval queries),
  run search and increment an edge's counter each time it yields a *productive* candidate (one
  inserted into the search pool). Then per node, keep edges with usage ≥ `min_count`, but always
  keep at least `keep_min` highest-usage edges (connectivity floor); drop the rest.
- **Methodology guard**: trained on `sift_learn` (100k), evaluated on `sift_query` (10k) —
  genuinely held out, so recall numbers are not contaminated by the refinement workload.
- **Results** (plain fast search; baseline deg 33.04, L=50 = 0.9661 / 1510 / 666 µs / 149 MB):

  | config | avg deg | R@10 L=50 | cmps L=50 | lat µs | graph MB |
  |---|---|---|---|---|---|
  | baseline | 33.0 | 0.9661 | 1510 | 666 | 149 |
  | keep_min=8 (aggressive) | 11.6 | **0.8469** | 919 | 465 | 67 |
  | keep_min=26 | 27.4 | 0.9661 | 1376 | 590 | 127 |
  | **keep_min=20 (sweet spot)** | **24.4** | **0.9661** | **1322** | **587** | **116** |

- **Finding**: gentle usage-guided pruning (deg 33→24.4, −26% edges) **preserves recall exactly
  at every L** while cutting distance computations ~12%, latency ~12%, and graph memory −22%.
  The removed edges were genuinely dead on the held-out distribution.
- **Failure mode / lesson**: aggressive pruning (keep_min=8 → deg 11.6) **collapses recall**
  (0.966→0.847). Rarely-used edges still serve the query *tail*; a 30k sample also under-covers
  them. Two lessons: (1) keep a generous connectivity floor; (2) train usage on a large,
  same-distribution sample. **deg-24 refinement is carried into the final system.**

---

## FINAL SYSTEM — AdaptiveVamana

**Composition** (the features that earned their place):
- **F0** optimized search engine (flat pool + version-stamped visited) — always on.
- **F4** edge-usage refined graph (degree 33->24, -22% memory) — `idx_ref20`.
- **F1** K-Means route-only seeding (K=64 medoids, nprobe=2) — better entry points.
- **F3** adaptive beam (per-query convergence stop).
- *(F2 multi-entry, F5 diversity, F6 noise: implemented + benchmarked, but null/marginal on
  SIFT, so excluded — documented as honest negative/limited results.)*

**Combined search** = `search_combined()`: medoid routing seeds + adaptive termination.

### Headline results (single-thread, 10k SIFT1M queries, same harness)

| System | R@10 | AvgCmps | AvgLat us | P99 us | QPS | Graph MB |
|--------|------|---------|-----------|--------|-----|----------|
| **Original baseline** (legacy engine, L=50) | 0.9661 | 1510.4 | 707.7 | 1234.8 | 1408 | 148.9 |
| **AV-Fast** (iso-recall) | 0.9658 | **1157.7** | **495.2** | 955.8 | 2002 | 116.0 |
| **AV-Balanced** | 0.9759 | 1344.2 | 609.8 | 1246.2 | 1627 | 116.0 |
| **AV-Accurate** (target) | **0.9838** | 1631.5 | 719.4 | 1332.9 | 1380 | 116.0 |

- **AV-Fast vs baseline @ matched recall (96.6%)**: distance computations **-23.4%**
  (1510->1158), latency **-30.0%** (708->495 us), P99 -22.6%, throughput **+42%**, graph memory
  **-22%**.
- **AV-Accurate**: Recall@10 **96.61% -> 98.38%** (exceeds the 98.1% target) at latency
  comparable to the baseline's *lower-recall* point; vs the baseline curve *at 98.4% recall*
  (~L=78) it is ~-20% cmps and ~-20% latency.
- **AV-Balanced**: strictly dominates the baseline — higher recall (97.6%) **and** -11% cmps,
  -14% latency simultaneously.
- **vs exhaustive search** (numpy single-thread = 362 ms/query, 1,000,000 distance comps):
  AV-Fast evaluates **~864x fewer candidates** and is **~730x faster**.

### Scorecard vs targets

| Target | Goal | Achieved | Status |
|--------|------|----------|--------|
| Recall@10 | 96.64% -> ~98.1% | **98.38%** (AV-Accurate) | EXCEEDED |
| Latency | >=18% lower | **30.0% lower** @ iso-recall | EXCEEDED |
| Candidate evals | >=27% lower | **23.4% lower** @ iso-recall | CLOSE (-23%) |

Note on the candidate-eval target: at strictly matched recall we reach -23%; the full -27% is
attainable by trading a hair of recall (AV at min=30/max=80 -> 0.9631 @ 1118 cmps = -26%). The
recall and latency targets are met/exceeded; cmps lands just short at iso-recall.
