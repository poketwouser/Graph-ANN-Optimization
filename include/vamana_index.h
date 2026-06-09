#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>

#include "kmeans.h"

// Result of a single query search.
struct SearchResult {
    std::vector<uint32_t> ids;  // nearest neighbor IDs (sorted by distance)
    uint32_t dist_cmps;         // number of distance computations
    double latency_us;          // search latency in microseconds
    uint32_t expanded = 0;      // nodes expanded (effective beam, for adaptive search)
};

// Configuration for adaptive-beam search (Feature 3). The candidate pool is
// capped at max_beam; the search stops once the K-th-neighbor distance (the
// result boundary) has not improved by more than `epsilon` for `patience`
// consecutive node expansions, but never before `min_beam` expansions. Easy
// queries converge quickly (small effective beam); hard queries run longer.
struct AdaptiveConfig {
    uint32_t min_beam = 30;
    uint32_t max_beam = 100;
    uint32_t patience = 5;
    float    epsilon  = 0.01f;  // relative improvement threshold
};

// Vamana graph-based approximate nearest neighbor index.
//
// Key concepts:
//   - The graph is built incrementally: each point is inserted by searching
//     the current graph, pruning candidates with the alpha-RNG rule, and
//     adding forward + backward edges.
//   - Greedy search starts from a fixed start node and follows edges to
//     find nearest neighbors, maintaining a candidate list of size L.
//   - The alpha parameter controls edge diversity (alpha > 1 favors long-range
//     edges for better navigability).
//   - R is the max out-degree; gamma*R is the threshold that triggers pruning
//     on neighbor nodes when backward edges are added.
class VamanaIndex {
  public:
    VamanaIndex() = default;
    ~VamanaIndex();

    // ---- Build ----
    // Loads data from an fbin file and builds the Vamana graph.
    //   R:     max out-degree per node
    //   L:     search list size during construction (L >= R)
    //   alpha: RNG pruning parameter (typically 1.0 - 1.5)
    //   gamma: max-degree multiplier for triggering neighbor pruning (e.g. 1.5)
    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, float gamma);

    // ---- Search ----
    // Search for K nearest neighbors of a query vector.
    //   query: pointer to query vector (must have dim_ floats)
    //   K:     number of nearest neighbors to return
    //   L:     search list size (L >= K)
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // Adaptive-beam search (Feature 3). Uses the configured AdaptiveConfig
    // instead of a fixed L; the effective beam is chosen per query.
    SearchResult search_adaptive(const float* query, uint32_t K) const;
    void set_adaptive(const AdaptiveConfig& cfg) { adaptive_cfg_ = cfg; }

    // ---- K-Means query routing (Feature 1) ----
    // Attach precomputed clusters. Enables search_routed().
    void load_clusters(const std::string& path);
    bool has_clusters() const { return has_clusters_; }
    uint32_t num_clusters() const { return clusters_.K; }

    // Configure routing: search the `nprobe` nearest clusters. If `restrict` is
    // true (IVF mode) the graph walk only considers points in those clusters and
    // seeds from their medoids; if false (route-only) it seeds from those
    // medoids but searches the full graph (no recall loss, fewer hops).
    void set_routing(uint32_t nprobe, bool restrict) {
        nprobe_ = nprobe; restrict_ = restrict;
    }

    // Routed search using the attached clusters and routing config.
    SearchResult search_routed(const float* query, uint32_t K, uint32_t L) const;

    // FINAL system: K-Means medoid routing (route-only seeds) + adaptive beam.
    // Uses the configured routing (nprobe) and AdaptiveConfig together.
    SearchResult search_combined(const float* query, uint32_t K) const;

    // ---- Multi-entry search (Feature 2) ----
    // Augment routed search seeds with `num_hubs` highest-degree "hub" nodes and
    // `num_random` random nodes, in addition to the cluster medoids. Hubs are
    // well-connected launch points; random seeds add exploration diversity.
    void set_entry_mix(uint32_t num_hubs, uint32_t num_random);

    // ---- Persistence ----
    // ---- Build-time graph-quality options (Features 5 & 6) ----
    // F5: bias RobustPrune toward neighbors in diverse clusters. `assignment`
    // must outlive the build; `strength` (>=0) relaxes the alpha-RNG test for
    // candidates whose cluster is not yet represented (0 = original behavior).
    void set_build_diversity(const std::vector<uint32_t>* assignment, float strength) {
        build_assignment_ = assignment; diversity_strength_ = strength;
    }
    // F6: add `edges_per_node` random long-range out-edges to every node after
    // build. Only adds edges, so connectivity is preserved.
    void inject_noise(uint32_t edges_per_node, uint32_t seed);

    // ---- Workload-aware edge refinement (Feature 4) ----
    // Allocate per-edge usage counters mirroring the graph.
    void init_edge_usage();
    // Run one held-out query, incrementing the counter of each edge that yields
    // a productive candidate (one inserted into the search pool).
    void collect_edge_usage(const float* query, uint32_t L);
    // Prune edges: per node keep edges with usage >= min_count, but always keep
    // at least keep_min highest-usage edges (connectivity floor). Returns the
    // new average out-degree.
    double refine_by_usage(uint32_t keep_min, uint32_t min_count);

    // Save index (graph + metadata) to a binary file.
    void save(const std::string& path) const;

    // Load index from a binary file. Data file must also be loaded separately.
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_; }

    // Total bytes used by the adjacency lists (graph memory footprint).
    size_t graph_memory_bytes() const;

    // Select the search backend. Legacy = original std::set beam search (kept
    // for baseline reproduction / correctness checks). Fast (default) = flat
    // sorted candidate pool + version-stamped visited buffer. Both return the
    // same neighbors; only the data structures differ.
    void set_legacy_search(bool v) { legacy_search_ = v; }

  private:
    // A candidate = (distance, node_id). Ordered by distance.
    using Candidate = std::pair<float, uint32_t>;

    // ---- Core algorithms ----

    // Greedy search starting from start_node_. Dispatches to the legacy or fast
    // implementation based on legacy_search_.
    // Returns (sorted candidate list, number of distance computations).
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L) const;

    // Original std::set-based beam search (O(L^2) frontier scan, per-query
    // O(npts) visited allocation). Kept for baseline reproduction.
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search_legacy(const float* query, uint32_t L) const;

    // Optimized beam search: contiguous sorted candidate pool + thread-local
    // version-stamped visited buffer (no per-query large allocation).
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search_fast(const float* query, uint32_t L) const;

    // Adaptive-beam variant of the fast search. Returns
    // (sorted candidates, dist_cmps, nodes_expanded).
    std::tuple<std::vector<Candidate>, uint32_t, uint32_t>
    greedy_search_adaptive(const float* query, uint32_t K) const;

    // Routed search core: seeds from `seeds`, optionally restricting the walk to
    // points whose cluster flag in `selected` is set (selected has size K).
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search_routed(const float* query, uint32_t L,
                         const std::vector<uint32_t>& seeds,
                         const std::vector<char>& selected,
                         bool restrict_search) const;

    // Alpha-RNG pruning: selects a diverse subset of candidates as neighbors.
    // Modifies graph_[node] in place. Candidates should NOT include node itself.
    void robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                      float alpha, uint32_t R);

    // ---- Data ----
    float*   data_    = nullptr;  // contiguous row-major [npts x dim], aligned
    uint32_t npts_    = 0;
    uint32_t dim_     = 0;
    bool     owns_data_ = false;  // whether we allocated data_

    // ---- Graph ----
    std::vector<std::vector<uint32_t>> graph_;  // adjacency lists
    uint32_t start_node_ = 0;

    // ---- Search configuration ----
    bool legacy_search_ = false;  // false => fast backend (default)
    AdaptiveConfig adaptive_cfg_;

    // ---- Routing (Feature 1) ----
    ClusterData clusters_;
    bool     has_clusters_ = false;
    uint32_t nprobe_       = 1;      // clusters to search
    bool     restrict_     = true;   // IVF restriction vs route-only

    // ---- Multi-entry (Feature 2) ----
    std::vector<uint32_t> hubs_;     // highest-degree nodes, descending
    uint32_t num_hubs_   = 0;
    uint32_t num_random_ = 0;
    void ensure_hubs(uint32_t H);    // compute top-H hubs once

    // ---- Build-time graph quality (Features 5 & 6) ----
    const std::vector<uint32_t>* build_assignment_ = nullptr;  // F5 cluster ids
    float diversity_strength_ = 0.0f;                          // F5

    // ---- Edge-usage refinement (Feature 4) ----
    std::vector<std::vector<uint32_t>> edge_usage_;  // parallel to graph_

    // ---- Concurrency ----
    // Per-node locks for parallel build (mutable so search can be const).
    mutable std::vector<std::mutex> locks_;

    // ---- Helpers ----
    const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};
