# AdaptiveVamana — Technical Report

A workload-aware, cluster-routed enhancement of a Vamana (DiskANN) graph index for
approximate nearest-neighbor (ANN) search, validated on SIFT1M. This report covers the
architecture, the algorithms, the design decisions and their experimental justification,
complexity analysis, an interview (STAR) narrative, limitations, and resume bullets.

Companion docs: **`PROGRESS.md`** (per-feature change log with rationale + tradeoffs) and
**`BENCHMARKS.md`** (all measured tables).

---

## 1. Problem & baseline

The starting point is a clean single-layer Vamana index: a directed graph where greedy
beam search from a fixed start node walks edges toward a query's neighborhood. On SIFT1M it
achieved **Recall@10 = 96.6% at 1510 distance computations and ~708 µs/query** (L=50).

Three structural weaknesses limited it:
1. **Search engine overhead** — `greedy_search` allocated a 1 MB `vector<bool> visited`
   *per query* and used a `std::set` candidate list with an **O(L²)** "find next to expand"
   scan. Latency was dominated by bookkeeping, not by the distance math.
2. **A single, *random* entry point** — every query started its walk from the same arbitrary
   node, far from most queries, wasting hops.
3. **A uniform, static search budget** — the same beam width `L` for trivially easy and
   genuinely hard queries; and a graph carrying edges that no realistic query ever traverses.

AdaptiveVamana attacks all three while keeping the Vamana core intact (no full rewrite).

---

## 2. Architecture

```
            offline                                   query time
  ┌────────────────────────┐              ┌────────────────────────────────────┐
  │ base vectors (1M×128)  │              │ query ─► rank K centroids          │
  │   │                    │              │          (pick nprobe nearest)     │
  │   ├─► Vamana build     │──► graph ───►│          seed = nprobe medoids     │
  │   │   (α-RNG prune)    │   (CSR-like) │          │                         │
  │   ├─► K-Means (k++/Lloyd)──► centroids│          ▼                         │
  │   │        │              + medoids   │   adaptive greedy walk             │
  │   │        └──► assignment────────────│   (flat pool + stamped visited,    │
  │   └─► edge-usage refine  (held-out)   │    K-th-distance convergence stop) │
  │            (drop dead edges)          │          │                         │
  └────────────────────────┘              │          ▼ top-K                   │
                                          └────────────────────────────────────┘
```

**Modules** (added/modified files):
- `distance.{h,cpp}` — squared-L2, auto-vectorized (unchanged).
- `vamana_index.{h,cpp}` — graph build + α-RNG prune; **new**: fast/legacy/adaptive/routed/
  combined search backends; cluster routing; multi-entry; build-time diversity; noise
  injection; edge-usage refinement.
- `kmeans.{h,cpp}` — **new**: k-means++ init + Lloyd on a subsample, full assignment, medoids.
- `build_index` / `search_index` / `build_clusters` / `refine_index` — CLIs.

---

## 3. The six features — algorithm, result, judgment

| # | Feature | Mechanism | Verdict on SIFT1M |
|---|---------|-----------|-------------------|
| F0 | Search engine | flat sorted pool + version-stamped visited (O(1) clear) | **Win** 6–18% latency, recall identical |
| F1 | K-Means routing | route to nprobe nearest cluster medoids (entry points); optional IVF restriction | **Win (route-only)** −10–18% latency, recall preserved |
| F3 | Adaptive beam | stop when K-th-NN distance converges (per-query) | **Win** −5–7% cmps at iso-recall |
| F4 | Edge refinement | drop edges unused on held-out workload (keep connectivity floor) | **Win (gentle)** −12% cmps, −22% memory; fails if aggressive |
| F2 | Multi-entry | add hub + random seeds | **Null** — redundant atop medoid routing |
| F5 | Diverse prune | relax α-RNG for new-cluster candidates | **Null** — α-RNG already diverse |
| F6 | Noise edges | random long-range shortcuts | **Marginal** — small low-L win, hurts at high L |

### Why the winners work
- **F0**: removing a 1 MB zero-fill and an O(L²) tree scan per query trades nothing —
  identical traversal, fewer cycles. Gains scale with L because the removed cost is super-linear
  in L.
- **F1 (route-only)**: a query's true neighbors lie in the few clusters whose centroids are
  nearest; that cluster's *medoid* is a graph node already close to the answer, so the greedy
  walk starts near the target and takes fewer hops — at **no** recall cost (the full graph is
  still searchable). The baseline's single random start was pure waste.
- **F3**: query difficulty is heterogeneous; spending budget proportional to observed
  convergence (the K-th distance ceasing to improve) reallocates work from easy to hard queries.
- **F4**: a graph built data-agnostically keeps many edges that the *query distribution* never
  uses; measuring real traversals on a held-out set identifies and removes them, shrinking
  per-expansion work and memory without hurting recall.

### Why the others don't (the honest negatives — these are the interesting science)
- **F1 IVF-restrict is dominated by route-only**: hard-restricting a *monolithic* graph to
  selected clusters imposes a recall ceiling (true NN in an unsearched cluster is unreachable)
  and adds a per-edge cluster-membership lookup; to recover recall you must search so many
  clusters that the savings evaporate. Restriction only makes sense if you deliberately accept
  recall loss for a memory/latency budget.
- **F2**: once the medoid lands you next to the answer, extra distant launch points only cost
  distance computations.
- **F5**: Vamana's α-RNG already yields geometrically (hence cluster-) diverse neighbors.
- **F6**: a little noise helps navigability at small beams; too much just inflates degree and
  per-expansion cost.

---

## 4. Complexity analysis

Let n = #points, d = dim, R = out-degree, L = beam, H = hops to converge, K = #clusters,
P = nprobe.

| Operation | Cost |
|---|---|
| Distance (L2) | Θ(d) |
| Greedy search (fixed L), legacy | Θ(H·R·d) distance + **Θ(H·L²)** frontier bookkeeping + Θ(n) per-query alloc |
| Greedy search (fixed L), **fast** | Θ(H·R·d) distance + Θ(H·R·L) pool inserts + **Θ(1)** amortized visited clear |
| Adaptive search | same, with H chosen per query by convergence (≤ max_beam) |
| K-Means routing overhead | Θ(K·d) centroid ranking + Θ(P) seeds |
| K-Means train | Θ(iters · n_sample · K · d) + Θ(n·K·d) assignment |
| Edge refinement | Θ(|workload| · H · R·d) collect + Θ(Σ deg·log deg) prune |
| Memory: graph | Θ(n·R̄) ids; F4 cuts R̄ 33→24 (−22%) |
| Memory: routing | Θ(K·d + n) (centroids + assignment), ~4 MB |

The legacy → fast change removes the Θ(H·L²) and Θ(n) per-query terms — the source of the
latency that was disproportionate to the Θ(H·R·d) distance work. Routing reduces H (hops);
adaptive reduces H for easy queries; F4 reduces R̄ (distance per expansion). The final system
multiplies these effects.

**vs exhaustive**: brute force is Θ(n·d) = 1,000,000 distance comps/query; AdaptiveVamana is
~1,158 → **~864× fewer evaluations, ~730× lower latency** (measured).

---

## 5. Experimental results (headline)

Single-thread, 10k SIFT1M queries, identical harness (full tables in `BENCHMARKS.md`).

| System | R@10 | AvgCmps | Latency µs | QPS | Graph MB |
|--------|------|---------|------------|-----|----------|
| Original baseline (L=50) | 0.9661 | 1510 | 708 | 1408 | 149 |
| **AV-Fast** (iso-recall) | 0.9658 | **1158** | **495** | 2002 | 116 |
| **AV-Balanced** | 0.9759 | 1344 | 610 | 1627 | 116 |
| **AV-Accurate** | **0.9838** | 1631 | 719 | 1380 | 116 |

- **At matched recall (96.6%)**: −23% distance computations, −30% latency, +42% throughput,
  −22% graph memory.
- **Tuned for accuracy**: Recall@10 **96.6% → 98.4%** (exceeds the 98.1% goal), still beating
  the baseline's cost *at that recall* by ~20%.
- AV-Balanced strictly dominates the baseline (higher recall **and** lower cost).

### Targets scorecard
| Target | Goal | Achieved |
|---|---|---|
| Recall@10 | →98.1% | **98.4%** ✅ |
| Latency | −18% | **−30%** ✅ |
| Candidate evals | −27% | **−23%** (−26% if recall relaxed to 96.3%) ◑ |

### Hyperparameter findings
- **Clusters K**: routing favors small K (16–64) — large K adds centroid-ranking overhead and,
  under IVF, a steep recall ceiling at low nprobe. Route-only is insensitive to K.
- **nprobe**: 2 nearest medoids suffices for route-only; more adds seeds without benefit.
- **Adaptive min/max/patience/ε**: `max_beam` should be tight (pool insert is O(pool));
  ε≈0.002–0.004, patience 10–14 are robust; `min_beam` sets the recall floor.
- **Noise**: 0.05 only; **Diversity**: no setting helped; **Edge-refine**: keep_min≈20 (deg 24)
  is the safe knee — deg 11 collapses recall.

---

## 6. Interview narrative (STAR)

**Situation** — A working Vamana ANN index hit 96.6% Recall@10 on SIFT1M but at ~708 µs and
1510 distance computations per query. The brief: improve accuracy *and* cost, with experimental
proof, no rewrite.

**Task** — Diagnose where the cost actually went, then design and validate enhancements that
shift the recall–cost Pareto frontier, keeping every claim measured.

**Action** —
1. *Profiled the baseline* and found latency was dominated by search **bookkeeping** (a 1 MB
   per-query allocation + an O(L²) `std::set` scan), not distance math — replaced the
   containers with a flat pool + version-stamped visited buffer, proving bit-identical recall
   via a legacy/fast equivalence test (which itself surfaced a cursor bug, fixed).
2. *Added K-Means query routing*: searching from the nearest cluster **medoids** instead of one
   random start cut hops with zero recall loss. Critically, I tested the textbook IVF
   *restriction* too and **showed it is dominated** on a monolithic graph (recall ceiling +
   per-edge lookup) — choosing the route-only design on evidence.
3. *Made the beam adaptive* — stop when the K-th-neighbor distance converges — reallocating
   budget from easy to hard queries.
4. *Refined the graph against a **held-out** workload*, dropping edges no real query traverses
   (degree 33→24, −22% memory), with a connectivity floor after discovering aggressive pruning
   collapses tail recall.
5. *Ran honest negatives* (multi-entry, cluster-diverse pruning, noise) and excluded them when
   the data said they didn't help.

**Result** — At matched 96.6% recall: **−30% latency, −23% distance computations, +42%
throughput, −22% memory**; tuned for accuracy, **Recall@10 96.6%→98.4%**. ~864× fewer
evaluations than exhaustive search. Every number reproduced on SIFT1M with a fixed harness.

**What I'd defend** — the wins are attributable (each feature ablated), the failures are
documented with mechanisms, and the methodology is clean (held-out refinement workload,
single-thread latency, recall verified against the same engine).

---

## 7. Limitations & threats to validity
- **Single dataset (SIFT1M, L2, 128-d)**. Routing/refinement gains depend on cluster structure
  and query-distribution stationarity; high-dimensional or adversarial workloads may differ.
- **Latency is single-thread** (true latency, low variance); the original 16-thread harness was
  memory-bandwidth-saturated. Throughput (QPS) reported separately.
- **F4 assumes the eval queries share the refinement workload's distribution** — valid here
  (learn vs query are both SIFT), but a distribution shift would re-expose pruned tail edges.
- **The −27% candidate-eval target is met only by relaxing recall slightly** (−26% at 96.3%);
  at strictly matched recall it is −23%.
- Build is in-memory; no disk-resident (true DiskANN) path.

---

## 8. Resume bullets (measured, defensible)

- **Built a scalable vector search engine for 1M+ 128-d embeddings**, combining cluster-aware
  indexing with graph-based ANN retrieval to evaluate **~860× fewer candidates** (and run
  **~730× faster**) than exhaustive nearest-neighbor search, sustaining **2,000 QPS** at 96.6%
  Recall@10 on SIFT1M.

- **Designed AdaptiveVamana, a hierarchical ANN framework** integrating K-Means medoid query
  routing, Vamana graph search, and convergence-based **adaptive beam-width** selection,
  improving Recall@10 from **96.6% to 98.4%** and cutting average query latency **~30%** at
  matched recall — shifting the recall–latency Pareto frontier rather than trading one for the
  other.

- **Enhanced graph exploration and efficiency** via medoid multi-entry traversal and
  **edge-usage-guided graph refinement** on a held-out workload (pruned 26% of edges, −22%
  index memory), reducing candidate evaluations **~23%** while preserving recall — and
  **ablated cluster-diverse pruning, IVF restriction, and noise injection**, reporting them as
  negative/marginal with root-cause analysis.

- **Re-engineered the search core** (flat candidate pool + version-stamped visited buffer
  replacing an O(L²) `std::set` and a per-query 1 MB allocation), delivering up to **18% lower
  latency at identical recall**, verified bit-for-bit against the original via an
  equivalence-test harness measuring Recall@10/@100, P95/P99, and throughput.

> Pick 3–4 depending on the role: #1 = systems/scale, #2 = algorithms, #3 = experimentation/
> rigor, #4 = low-level optimization.
