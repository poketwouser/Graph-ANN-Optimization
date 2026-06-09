#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <tuple>
#include <limits>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        portable_free(data_);  // data_ came from aligned alloc (io_utils)
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search
// ============================================================================
// Beam search starting from start_node_. Maintains a candidate set of at most
// L nodes, always expanding the closest unvisited node. Returns when no
// unvisited candidates remain.
//
// Uses std::set<Candidate> as an ordered container — simple, correct, and
// easy for students to understand and modify.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    return legacy_search_ ? greedy_search_legacy(query, L)
                          : greedy_search_fast(query, L);
}

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search_legacy(const float* query, uint32_t L) const {
    // Candidate set: ordered by (distance, id). Bounded at size L.
    std::set<Candidate> candidate_set;
    // Track which nodes we've already expanded (visited).
    std::vector<bool> visited(npts_, false);

    uint32_t dist_cmps = 0;

    // Seed with start node
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = true;

    // Track which candidates have been expanded (their neighbors explored).
    // We iterate through candidate_set; entries before our "frontier" pointer
    // have been expanded. We use a simple approach: keep scanning from the
    // beginning of the set for the first un-expanded entry.
    std::set<uint32_t> expanded;

    while (true) {
        // Find closest candidate that hasn't been expanded yet
        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                best_node = id;
                break;
            }
        }
        if (best_node == UINT32_MAX)
            break;  // all candidates expanded

        expanded.insert(best_node);

        // Expand: evaluate all neighbors of best_node
        // Copy neighbor list under lock to avoid data race with parallel build
        // (another thread might push_back / reallocate graph_[best_node]).
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }
        for (uint32_t nbr : neighbors) {
            if (visited[nbr])
                continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            // Insert if candidate set isn't full or this is closer than worst
            if (candidate_set.size() < L) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
        }
    }

    // Convert to sorted vector
    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ============================================================================
// Fast Greedy Search
// ============================================================================
// Same algorithm as the legacy version (expand the closest unexpanded
// candidate, keep the L best), but with cache-friendly data structures:
//
//   * Candidate pool: a contiguous std::vector<Neighbor> kept sorted by
//     (distance, id). Inserting a new candidate is a bounded shift; finding the
//     next node to expand is an O(1) cursor advance rather than an O(L) scan of
//     a red-black tree (legacy was O(L^2) overall per query).
//
//   * Visited set: a thread-local version-stamped buffer. Instead of allocating
//     a fresh std::vector<bool>(npts_) (~1 MB) and a std::set<uint32_t> on every
//     query, we keep one uint32 tag per node and bump a generation counter to
//     "clear" it in O(1). Sized once, reused across all queries on the thread.
//
// Both backends return identical neighbor sets for the same graph and query.

namespace {

// Sorted-by-(dist,id) candidate, plus an "expanded" flag.
struct Neighbor {
    float    dist;
    uint32_t id;
    bool     expanded;
};

// Thread-local visited buffer with O(1) clear via generation stamping.
struct VisitedList {
    std::vector<uint32_t> tag;   // tag[i] == gen  <=>  node i visited this query
    uint32_t gen = 0;

    void ensure(size_t n) {
        if (tag.size() < n) { tag.assign(n, 0); gen = 0; }
    }
    void clear() {
        if (++gen == 0) {        // generation wrapped: do a real reset
            std::fill(tag.begin(), tag.end(), 0);
            gen = 1;
        }
    }
    bool test(uint32_t i) const { return tag[i] == gen; }
    void set(uint32_t i)        { tag[i] = gen; }
};

}  // namespace

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search_fast(const float* query, uint32_t L) const {
    static thread_local VisitedList visited;
    visited.ensure(npts_);
    visited.clear();

    std::vector<Neighbor> pool;
    pool.reserve(L + 1);

    uint32_t dist_cmps = 0;

    // Seed with start node.
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    pool.push_back({start_dist, start_node_, false});
    visited.set(start_node_);

    // Cursor: index of the closest candidate not yet expanded.
    uint32_t cur = 0;
    while (cur < pool.size()) {
        if (pool[cur].expanded) { cur++; continue; }
        pool[cur].expanded = true;
        uint32_t node = pool[cur].id;

        // Copy neighbors under lock (parallel build may mutate graph_[node]).
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[node]);
            neighbors = graph_[node];
        }

        uint32_t next_cur = pool.size();  // smallest insert position this round
        for (uint32_t nbr : neighbors) {
            if (visited.test(nbr)) continue;
            visited.set(nbr);

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            // Skip if pool is full and this is no better than the current worst.
            if (pool.size() >= L && d >= pool.back().dist)
                continue;

            // Insert keeping the pool sorted by (dist, id).
            Neighbor nn{d, nbr, false};
            auto it = std::lower_bound(
                pool.begin(), pool.end(), nn,
                [](const Neighbor& a, const Neighbor& b) {
                    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
                });
            uint32_t pos = (uint32_t)(it - pool.begin());
            pool.insert(it, nn);
            if (pool.size() > L) pool.pop_back();
            if (pos < next_cur) next_cur = pos;
        }

        // Move the cursor to the closest unexpanded node. If a new candidate
        // landed at or before the current position (next_cur <= cur), it is now
        // the closest unexpanded entry and must be expanded next; otherwise all
        // positions <= cur are expanded, so advance by one. Using '<=' (not '<')
        // is essential: a neighbor closer than the node we just expanded but not
        // closer than its predecessor inserts exactly at `cur` and would
        // otherwise be skipped.
        if (next_cur <= cur) cur = next_cur; else cur++;
    }

    std::vector<Candidate> results;
    results.reserve(pool.size());
    for (const auto& n : pool) results.push_back({n.dist, n.id});
    return {results, dist_cmps};
}

// ============================================================================
// Adaptive Greedy Search (Feature 3)
// ============================================================================
// Same traversal as greedy_search_fast, but the beam is chosen per query.
// The pool is capped at adaptive_cfg_.max_beam. We track the K-th-neighbor
// distance (the boundary of the result set) and stop once it has not improved
// by more than `epsilon` for `patience` consecutive expansions — but never
// before `min_beam` expansions. Easy queries (boundary stabilizes quickly)
// stop early with a small effective beam; hard queries keep exploring up to
// max_beam. Returns (candidates, dist_cmps, nodes_expanded).

std::tuple<std::vector<VamanaIndex::Candidate>, uint32_t, uint32_t>
VamanaIndex::greedy_search_adaptive(const float* query, uint32_t K) const {
    const uint32_t max_beam = std::max(adaptive_cfg_.max_beam, K);
    const uint32_t min_beam = std::min(adaptive_cfg_.min_beam, max_beam);
    const uint32_t patience = adaptive_cfg_.patience;
    const float    eps      = adaptive_cfg_.epsilon;

    static thread_local VisitedList visited;
    visited.ensure(npts_);
    visited.clear();

    std::vector<Neighbor> pool;
    pool.reserve(max_beam + 1);

    uint32_t dist_cmps = 0;
    uint32_t expanded  = 0;

    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    pool.push_back({start_dist, start_node_, false});
    visited.set(start_node_);

    float kth_best = std::numeric_limits<float>::max();  // result-boundary dist
    uint32_t stall = 0;

    uint32_t cur = 0;
    while (cur < pool.size()) {
        if (pool[cur].expanded) { cur++; continue; }
        pool[cur].expanded = true;
        uint32_t node = pool[cur].id;
        expanded++;

        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[node]);
            neighbors = graph_[node];
        }

        uint32_t next_cur = pool.size();
        for (uint32_t nbr : neighbors) {
            if (visited.test(nbr)) continue;
            visited.set(nbr);

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            if (pool.size() >= max_beam && d >= pool.back().dist)
                continue;

            Neighbor nn{d, nbr, false};
            auto it = std::lower_bound(
                pool.begin(), pool.end(), nn,
                [](const Neighbor& a, const Neighbor& b) {
                    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
                });
            uint32_t pos = (uint32_t)(it - pool.begin());
            pool.insert(it, nn);
            if (pool.size() > max_beam) pool.pop_back();
            if (pos < next_cur) next_cur = pos;
        }

        // Convergence test on the result boundary (K-th distance).
        float cur_kth = (pool.size() >= K) ? pool[K - 1].dist : pool.back().dist;
        if (cur_kth < kth_best * (1.0f - eps)) {
            kth_best = cur_kth;
            stall = 0;
        } else {
            stall++;
        }

        // Adaptive stop: enough work done and the boundary has converged.
        if (expanded >= min_beam && stall >= patience)
            break;

        if (next_cur <= cur) cur = next_cur; else cur++;
    }

    std::vector<Candidate> results;
    results.reserve(pool.size());
    for (const auto& n : pool) results.push_back({n.dist, n.id});
    return {results, dist_cmps, expanded};
}

// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================
// Given a node and a set of candidates, greedily select neighbors that are
// "diverse" — a candidate c is added only if it's not too close to any
// already-selected neighbor (within a factor of alpha).
//
// Formally: add c if for ALL already-chosen neighbors n:
//     dist(node, c) <= alpha * dist(c, n)
//
// This ensures good graph navigability by keeping some long-range edges
// (alpha > 1 makes it easier for a candidate to survive pruning).

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    // Remove self from candidates if present
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Sort by distance to node (ascending)
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    // F5: track clusters already represented among the selected neighbors so we
    // can favor candidates from new clusters (relaxed alpha-RNG threshold).
    const bool diverse = (build_assignment_ != nullptr && diversity_strength_ > 0.0f);
    std::vector<uint32_t> covered;  // small (<= R); linear scan is fine

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R)
            break;

        // Effective alpha: relaxed for candidates whose cluster is not yet
        // represented, so the neighbor set spans more clusters.
        float eff_alpha = alpha;
        uint32_t cand_cluster = 0;
        if (diverse) {
            cand_cluster = (*build_assignment_)[cand_id];
            bool seen = false;
            for (uint32_t c : covered) if (c == cand_cluster) { seen = true; break; }
            if (!seen) eff_alpha = alpha * (1.0f + diversity_strength_);
        }

        // Check alpha-RNG condition against all already-selected neighbors
        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > eff_alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }

        if (keep) {
            new_neighbors.push_back(cand_id);
            if (diverse) {
                bool seen = false;
                for (uint32_t c : covered) if (c == cand_cluster) { seen = true; break; }
                if (!seen) covered.push_back(cand_cluster);
            }
        }
    }

    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    // --- Load data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R) {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    // --- Initialize empty graph and per-node locks ---
    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // --- Pick random start node ---
    std::mt19937 rng(42);  // fixed seed for reproducibility
    start_node_ = rng() % npts_;
    std::cout << "  Start node: " << start_node_ << std::endl;

    // --- Create random insertion order ---
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // --- Build graph: parallel insertion with per-node locking ---
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L=" << L
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", gammaR=" << gamma_R << ")..." << std::endl;

    Timer build_timer;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // Step 1: Search for this point in the current graph to find candidates
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L);

        // Step 2: Prune candidates to get this point's neighbors
        // We don't need to lock graph_[point] here because each point appears
        // exactly once in the permutation — only this thread writes to it now.
        robust_prune(point, candidates, alpha, R);

        // Step 3: Add backward edges from each new neighbor back to this point
        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);

            // Add backward edge
            graph_[nbr].push_back(point);

            // Step 4: If neighbor's degree exceeds gamma*R, prune its neighborhood
            if (graph_[nbr].size() > gamma_R) {
                // Build candidate list from current neighbors of nbr
                std::vector<Candidate> nbr_candidates;
                nbr_candidates.reserve(graph_[nbr].size());
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_candidates.push_back({d, nn});
                }
                robust_prune(nbr, nbr_candidates, alpha, R);
            }
        }

        // Progress reporting (from one thread only)
        if (idx % 10000 == 0) {
            #pragma omp critical
            {
                std::cout << "\r  Inserted " << idx << " / " << npts_
                          << " points" << std::flush;
            }
        }
    }

    double build_time = build_timer.elapsed_seconds();

    // Compute average degree
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();
    double avg_degree = (double)total_edges / npts_;

    std::cout << "\n  Build complete in " << build_time << " seconds."
              << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;
}

// ============================================================================
// Selective Noise Injection (Feature 6)
// ============================================================================
// Add `edges_per_node` random long-range out-edges to every node. These act as
// small-world shortcuts that can shorten paths to distant regions. Edges are
// only added (deduped against existing neighbors and self), so connectivity is
// preserved. Done as a post-build pass.

void VamanaIndex::inject_noise(uint32_t edges_per_node, uint32_t seed) {
    if (edges_per_node == 0) return;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> pick(0, npts_ - 1);

    size_t added = 0;
    for (uint32_t u = 0; u < npts_; u++) {
        for (uint32_t k = 0; k < edges_per_node; k++) {
            uint32_t v = pick(rng);
            if (v == u) continue;
            // Dedup against existing neighbors (degree is small).
            bool exists = false;
            for (uint32_t w : graph_[u]) if (w == v) { exists = true; break; }
            if (exists) continue;
            graph_[u].push_back(v);
            added++;
        }
    }

    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++) total_edges += graph_[i].size();
    std::cout << "  Noise injection: +" << added << " random edges, new avg degree "
              << (double)total_edges / npts_ << std::endl;
}

// ============================================================================
// Search
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;

    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L);
    double latency = t.elapsed_us();

    // Return top-K results
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

SearchResult VamanaIndex::search_adaptive(const float* query, uint32_t K) const {
    Timer t;
    auto [candidates, dist_cmps, expanded] = greedy_search_adaptive(query, K);
    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.expanded = expanded;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Edge-Usage-Guided Refinement (Feature 4)
// ============================================================================
// Collect, over a held-out workload, how often each edge yields a *productive*
// candidate (one good enough to enter the search pool). Edges that never do are
// dead weight: pruning them shrinks the graph and cuts per-expansion distance
// computations without hurting recall on a same-distribution query set. A
// per-node connectivity floor (keep_min) prevents disconnection.

void VamanaIndex::init_edge_usage() {
    edge_usage_.resize(npts_);
    for (uint32_t u = 0; u < npts_; u++)
        edge_usage_[u].assign(graph_[u].size(), 0u);
}

void VamanaIndex::collect_edge_usage(const float* query, uint32_t L) {
    static thread_local VisitedList visited;
    visited.ensure(npts_);
    visited.clear();

    std::vector<Neighbor> pool;
    pool.reserve(L + 1);

    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    pool.push_back({start_dist, start_node_, false});
    visited.set(start_node_);

    uint32_t cur = 0;
    while (cur < pool.size()) {
        if (pool[cur].expanded) { cur++; continue; }
        pool[cur].expanded = true;
        uint32_t node = pool[cur].id;

        const std::vector<uint32_t>& neighbors = graph_[node];
        uint32_t next_cur = pool.size();
        for (uint32_t j = 0; j < neighbors.size(); j++) {
            uint32_t nbr = neighbors[j];
            if (visited.test(nbr)) continue;
            visited.set(nbr);

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            if (pool.size() >= L && d >= pool.back().dist) continue;

            // Productive edge: it discovered a pool-worthy candidate.
            edge_usage_[node][j]++;

            Neighbor nn{d, nbr, false};
            auto it = std::lower_bound(pool.begin(), pool.end(), nn,
                [](const Neighbor& a, const Neighbor& b) {
                    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
                });
            uint32_t pos = (uint32_t)(it - pool.begin());
            pool.insert(it, nn);
            if (pool.size() > L) pool.pop_back();
            if (pos < next_cur) next_cur = pos;
        }
        if (next_cur <= cur) cur = next_cur; else cur++;
    }
}

double VamanaIndex::refine_by_usage(uint32_t keep_min, uint32_t min_count) {
    size_t total = 0, removed = 0;
    for (uint32_t u = 0; u < npts_; u++) {
        const auto& nbrs = graph_[u];
        const auto& use  = edge_usage_[u];
        uint32_t deg = (uint32_t)nbrs.size();

        // Order edge indices by usage (descending).
        std::vector<uint32_t> order(deg);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) { return use[a] > use[b]; });

        std::vector<uint32_t> kept;
        kept.reserve(deg);
        for (uint32_t rank = 0; rank < deg; rank++) {
            uint32_t j = order[rank];
            if (rank < keep_min || use[j] >= min_count)
                kept.push_back(nbrs[j]);
        }
        removed += deg - kept.size();
        total += deg;
        graph_[u] = std::move(kept);
    }
    size_t new_total = total - removed;
    std::cout << "  Refinement: removed " << removed << " / " << total
              << " edges (" << (100.0 * removed / total) << "%)" << std::endl;
    return (double)new_total / npts_;
}

// ============================================================================
// K-Means Query Routing (Feature 1)
// ============================================================================

void VamanaIndex::load_clusters(const std::string& path) {
    clusters_ = ::load_clusters(path);
    if (clusters_.npts != npts_ || clusters_.dim != dim_)
        throw std::runtime_error("Clusters/index mismatch (npts or dim)");
    has_clusters_ = true;
}

// Compute the H highest out-degree nodes (graph hubs) once. These are
// well-connected launch points reachable to many regions in few hops.
void VamanaIndex::ensure_hubs(uint32_t H) {
    if (hubs_.size() >= H) return;
    std::vector<uint32_t> idx(npts_);
    std::iota(idx.begin(), idx.end(), 0u);
    std::partial_sort(idx.begin(), idx.begin() + H, idx.end(),
        [this](uint32_t a, uint32_t b) {
            return graph_[a].size() > graph_[b].size();
        });
    hubs_.assign(idx.begin(), idx.begin() + H);
}

void VamanaIndex::set_entry_mix(uint32_t num_hubs, uint32_t num_random) {
    num_hubs_ = num_hubs;
    num_random_ = num_random;
    if (num_hubs > 0) ensure_hubs(num_hubs);
}

// Greedy search seeded from `seeds`, optionally restricted to selected clusters.
std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search_routed(const float* query, uint32_t L,
                                  const std::vector<uint32_t>& seeds,
                                  const std::vector<char>& selected,
                                  bool restrict_search) const {
    static thread_local VisitedList visited;
    visited.ensure(npts_);
    visited.clear();

    const uint32_t* assign = clusters_.assignment.data();

    std::vector<Neighbor> pool;
    pool.reserve(L + 1);
    uint32_t dist_cmps = 0;

    // Seed from all entry points (cluster medoids).
    for (uint32_t s : seeds) {
        if (visited.test(s)) continue;
        visited.set(s);
        float d = compute_l2sq(query, get_vector(s), dim_);
        dist_cmps++;
        Neighbor nn{d, s, false};
        auto it = std::lower_bound(pool.begin(), pool.end(), nn,
            [](const Neighbor& a, const Neighbor& b) {
                return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
            });
        pool.insert(it, nn);
        if (pool.size() > L) pool.pop_back();
    }

    uint32_t cur = 0;
    while (cur < pool.size()) {
        if (pool[cur].expanded) { cur++; continue; }
        pool[cur].expanded = true;
        uint32_t node = pool[cur].id;

        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[node]);
            neighbors = graph_[node];
        }

        uint32_t next_cur = pool.size();
        for (uint32_t nbr : neighbors) {
            // IVF restriction: ignore points outside the selected clusters.
            if (restrict_search && !selected[assign[nbr]]) continue;
            if (visited.test(nbr)) continue;
            visited.set(nbr);

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            if (pool.size() >= L && d >= pool.back().dist) continue;

            Neighbor nn{d, nbr, false};
            auto it = std::lower_bound(pool.begin(), pool.end(), nn,
                [](const Neighbor& a, const Neighbor& b) {
                    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
                });
            uint32_t pos = (uint32_t)(it - pool.begin());
            pool.insert(it, nn);
            if (pool.size() > L) pool.pop_back();
            if (pos < next_cur) next_cur = pos;
        }

        if (next_cur <= cur) cur = next_cur; else cur++;
    }

    std::vector<Candidate> results;
    results.reserve(pool.size());
    for (const auto& n : pool) results.push_back({n.dist, n.id});
    return {results, dist_cmps};
}

// FINAL combined search: route-only medoid seeding + adaptive beam termination.
SearchResult VamanaIndex::search_combined(const float* query, uint32_t K) const {
    Timer t;
    const uint32_t Kc = clusters_.K;
    const uint32_t nprobe = std::min(nprobe_, Kc);
    const uint32_t max_beam = std::max(adaptive_cfg_.max_beam, K);
    const uint32_t min_beam = std::min(adaptive_cfg_.min_beam, max_beam);
    const uint32_t patience = adaptive_cfg_.patience;
    const float    eps      = adaptive_cfg_.epsilon;

    // 1. Rank clusters; collect nprobe nearest medoids as seeds (route-only).
    std::vector<Candidate> cdist(Kc);
    for (uint32_t c = 0; c < Kc; c++)
        cdist[c] = {compute_l2sq(query, clusters_.centroid(c), dim_), c};
    std::partial_sort(cdist.begin(), cdist.begin() + nprobe, cdist.end());

    static thread_local VisitedList visited;
    visited.ensure(npts_);
    visited.clear();

    std::vector<Neighbor> pool;
    pool.reserve(max_beam + 1);
    uint32_t dist_cmps = Kc;  // centroid comparisons

    auto cmp = [](const Neighbor& a, const Neighbor& b) {
        return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
    };
    for (uint32_t i = 0; i < nprobe; i++) {
        uint32_t s = clusters_.medoids[cdist[i].second];
        if (visited.test(s)) continue;
        visited.set(s);
        float d = compute_l2sq(query, get_vector(s), dim_);
        dist_cmps++;
        Neighbor nn{d, s, false};
        pool.insert(std::lower_bound(pool.begin(), pool.end(), nn, cmp), nn);
        if (pool.size() > max_beam) pool.pop_back();
    }

    // 2. Adaptive greedy walk.
    float kth_best = std::numeric_limits<float>::max();
    uint32_t stall = 0, expanded = 0, cur = 0;
    while (cur < pool.size()) {
        if (pool[cur].expanded) { cur++; continue; }
        pool[cur].expanded = true;
        uint32_t node = pool[cur].id;
        expanded++;

        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[node]);
            neighbors = graph_[node];
        }

        uint32_t next_cur = pool.size();
        for (uint32_t nbr : neighbors) {
            if (visited.test(nbr)) continue;
            visited.set(nbr);
            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;
            if (pool.size() >= max_beam && d >= pool.back().dist) continue;
            Neighbor nn{d, nbr, false};
            auto it = std::lower_bound(pool.begin(), pool.end(), nn, cmp);
            uint32_t pos = (uint32_t)(it - pool.begin());
            pool.insert(it, nn);
            if (pool.size() > max_beam) pool.pop_back();
            if (pos < next_cur) next_cur = pos;
        }

        float cur_kth = (pool.size() >= K) ? pool[K - 1].dist : pool.back().dist;
        if (cur_kth < kth_best * (1.0f - eps)) { kth_best = cur_kth; stall = 0; }
        else stall++;
        if (expanded >= min_beam && stall >= patience) break;

        if (next_cur <= cur) cur = next_cur; else cur++;
    }

    double latency = t.elapsed_us();
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.expanded = expanded;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < pool.size(); i++)
        result.ids.push_back(pool[i].id);
    return result;
}

SearchResult VamanaIndex::search_routed(const float* query, uint32_t K,
                                        uint32_t L) const {
    if (L < K) L = K;
    Timer t;

    const uint32_t Kc = clusters_.K;
    const uint32_t nprobe = std::min(nprobe_, Kc);

    // 1. Rank clusters by distance to their centroids (counts as work).
    std::vector<Candidate> cdist(Kc);
    for (uint32_t c = 0; c < Kc; c++)
        cdist[c] = {compute_l2sq(query, clusters_.centroid(c), dim_), c};
    std::partial_sort(cdist.begin(), cdist.begin() + nprobe, cdist.end());

    // 2. Mark selected clusters and collect their medoid entry points.
    static thread_local std::vector<char> selected;
    selected.assign(Kc, 0);
    std::vector<uint32_t> seeds;
    seeds.reserve(nprobe + num_hubs_ + num_random_);
    for (uint32_t i = 0; i < nprobe; i++) {
        uint32_t c = cdist[i].second;
        selected[c] = 1;
        seeds.push_back(clusters_.medoids[c]);
    }

    // 2b. Multi-entry (Feature 2): add hub and random launch points.
    for (uint32_t i = 0; i < num_hubs_ && i < hubs_.size(); i++)
        seeds.push_back(hubs_[i]);
    if (num_random_ > 0) {
        static thread_local std::mt19937 rng(std::random_device{}());
        for (uint32_t i = 0; i < num_random_; i++)
            seeds.push_back(rng() % npts_);
    }

    // 3. Routed graph search.
    auto [candidates, gcmps] =
        greedy_search_routed(query, L, seeds, selected, restrict_);

    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps = gcmps + Kc;  // include centroid comparisons
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++)
        result.ids.push_back(candidates[i].second);
    return result;
}

// Sum of adjacency-list storage: the id array of each node plus the per-vector
// std::vector bookkeeping. This is the graph's resident footprint (vectors are
// stored separately in data_).
size_t VamanaIndex::graph_memory_bytes() const {
    size_t bytes = sizeof(graph_);
    for (const auto& adj : graph_)
        bytes += sizeof(adj) + adj.capacity() * sizeof(uint32_t);
    return bytes;
}

// ============================================================================
// Save / Load
// ============================================================================
// Binary format:
//   [uint32] npts
//   [uint32] dim
//   [uint32] start_node
//   For each node i in [0, npts):
//     [uint32] degree
//     [uint32 * degree] neighbor IDs

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0) {
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts), 4);
    in.read(reinterpret_cast<char*>(&file_dim), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0) {
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}
