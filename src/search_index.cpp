#include "vamana_index.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index <index_path>"
              << " --data <fbin_path>"
              << " --queries <query_fbin_path>"
              << " --gt <ground_truth_ibin_path>"
              << " --K <num_neighbors>"
              << " --L <comma_separated_L_values>"
              << std::endl;
}

// Parse comma-separated L values like "10,20,50,100"
static std::vector<uint32_t> parse_L_values(const std::string& s) {
    std::vector<uint32_t> values;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back(std::atoi(token.c_str()));
    }
    std::sort(values.begin(), values.end());
    return values;
}

// Compute recall@K: fraction of true top-K neighbors found in result
static double compute_recall(const std::vector<uint32_t>& result,
                             const uint32_t* gt, uint32_t K) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < K && i < result.size(); i++) {
        for (uint32_t j = 0; j < K; j++) {
            if (result[i] == gt[j]) {
                found++;
                break;
            }
        }
    }
    return (double)found / K;
}

int main(int argc, char** argv) {
    std::string index_path, data_path, query_path, gt_path, L_str;
    uint32_t K = 10;
    bool legacy_search = false;  // default: fast search backend

    // Adaptive-beam (Feature 3) options.
    bool adaptive = false;
    bool final_mode = false;  // routing + adaptive combined (final system)
    AdaptiveConfig acfg;  // defaults from header

    // Routing (Feature 1) options.
    std::string clusters_path;
    uint32_t clusters_to_search = 1;
    bool route_only = false;  // false => IVF restriction (default when routing)
    uint32_t num_hubs = 0, num_random = 0;  // Feature 2 multi-entry

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc)    index_path = argv[++i];
        else if (arg == "--data" && i + 1 < argc) data_path = argv[++i];
        else if (arg == "--queries" && i + 1 < argc) query_path = argv[++i];
        else if (arg == "--gt" && i + 1 < argc)   gt_path = argv[++i];
        else if (arg == "--K" && i + 1 < argc)    K = std::atoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc)    L_str = argv[++i];
        else if (arg == "--legacy-search")        legacy_search = true;
        else if (arg == "--adaptive")             adaptive = true;
        else if (arg == "--final")                final_mode = true;
        else if (arg == "--min-beam" && i + 1 < argc) acfg.min_beam = std::atoi(argv[++i]);
        else if (arg == "--max-beam" && i + 1 < argc) acfg.max_beam = std::atoi(argv[++i]);
        else if (arg == "--patience" && i + 1 < argc) acfg.patience = std::atoi(argv[++i]);
        else if (arg == "--epsilon"  && i + 1 < argc) acfg.epsilon  = std::atof(argv[++i]);
        else if (arg == "--clusters" && i + 1 < argc) clusters_path = argv[++i];
        else if (arg == "--clusters-to-search" && i + 1 < argc) clusters_to_search = std::atoi(argv[++i]);
        else if (arg == "--route-only")              route_only = true;
        else if (arg == "--hubs" && i + 1 < argc)    num_hubs = std::atoi(argv[++i]);
        else if (arg == "--random-entries" && i + 1 < argc) num_random = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (index_path.empty() || data_path.empty() || query_path.empty() ||
        gt_path.empty() || (L_str.empty() && !adaptive && !final_mode)) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<uint32_t> L_values;
    if (!adaptive && !final_mode) {
        L_values = parse_L_values(L_str);
        if (L_values.empty()) {
            std::cerr << "Error: no L values provided." << std::endl;
            return 1;
        }
    }

    // --- Load index ---
    std::cout << "Loading index..." << std::endl;
    VamanaIndex index;
    index.load(index_path, data_path);
    index.set_legacy_search(legacy_search);
    std::cout << "Search backend: " << (legacy_search ? "LEGACY (std::set)"
                                                       : "FAST (flat pool)") << std::endl;

    // Optional K-Means routing (Feature 1).
    bool routing = !clusters_path.empty();
    if (routing) {
        index.load_clusters(clusters_path);
        index.set_routing(clusters_to_search, !route_only);
        index.set_entry_mix(num_hubs, num_random);
        std::cout << "Routing: " << index.num_clusters() << " clusters, searching "
                  << clusters_to_search << " nearest, mode="
                  << (route_only ? "ROUTE-ONLY (full graph)" : "IVF-RESTRICT")
                  << "; entry-mix hubs=" << num_hubs << " random=" << num_random
                  << std::endl;
    }

    // --- Load queries ---
    std::cout << "Loading queries from " << query_path << "..." << std::endl;
    FloatMatrix queries = load_fbin(query_path);
    std::cout << "  Queries: " << queries.npts << " x " << queries.dims << std::endl;

    if (queries.dims != index.get_dim()) {
        std::cerr << "Error: query dimension (" << queries.dims
                  << ") != index dimension (" << index.get_dim() << ")" << std::endl;
        return 1;
    }

    // --- Load ground truth ---
    std::cout << "Loading ground truth from " << gt_path << "..." << std::endl;
    IntMatrix gt = load_ibin(gt_path);
    std::cout << "  Ground truth: " << gt.npts << " x " << gt.dims << std::endl;

    if (gt.npts != queries.npts) {
        std::cerr << "Error: ground truth rows (" << gt.npts
                  << ") != number of queries (" << queries.npts << ")" << std::endl;
        return 1;
    }
    if (gt.dims < K) {
        std::cerr << "Warning: ground truth has " << gt.dims
                  << " neighbors per query but K=" << K << std::endl;
        K = gt.dims;
    }

    uint32_t nq = queries.npts;

    // Recall cutoffs. recall@100 is only fully meaningful when L >= 100 (you
    // cannot return 100 neighbors from a beam smaller than 100).
    const uint32_t R10 = std::min<uint32_t>(10, gt.dims);
    const uint32_t R100 = std::min<uint32_t>(100, gt.dims);

    // --- Index memory footprint ---
    double graph_mb = index.graph_memory_bytes() / (1024.0 * 1024.0);
    double data_mb  = (double)index.get_npts() * index.get_dim() * sizeof(float)
                      / (1024.0 * 1024.0);
    std::cout << "\nIndex memory: graph " << std::fixed << std::setprecision(1)
              << graph_mb << " MB + vectors " << data_mb << " MB = "
              << (graph_mb + data_mb) << " MB" << std::endl;

    // --- Adaptive-beam (F3) or FINAL combined (routing + adaptive) mode ---
    if (adaptive || final_mode) {
        index.set_adaptive(acfg);
        std::cout << "\n=== " << (final_mode ? "FINAL (routing + adaptive)"
                                             : "Adaptive-Beam") << " Search ===" << std::endl;
        std::cout << "  min_beam=" << acfg.min_beam << " max_beam=" << acfg.max_beam
                  << " patience=" << acfg.patience << " epsilon="
                  << std::setprecision(4) << acfg.epsilon << std::endl;

        std::vector<double> recall10(nq), recall100(nq);
        std::vector<uint32_t> dist_cmps(nq);
        std::vector<double> latencies(nq);
        std::vector<uint32_t> beams(nq);
        const uint32_t Kret = std::min<uint32_t>(R100, acfg.max_beam);

        Timer wall;
        #pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t q = 0; q < nq; q++) {
            SearchResult res = final_mode ? index.search_combined(queries.row(q), Kret)
                                          : index.search_adaptive(queries.row(q), Kret);
            recall10[q]  = compute_recall(res.ids, gt.row(q), R10);
            recall100[q] = compute_recall(res.ids, gt.row(q), R100);
            dist_cmps[q] = res.dist_cmps;
            latencies[q] = res.latency_us;
            beams[q]     = res.expanded;
        }
        double qps = nq / wall.elapsed_seconds();

        double avg_r10  = std::accumulate(recall10.begin(),  recall10.end(),  0.0) / nq;
        double avg_r100 = std::accumulate(recall100.begin(), recall100.end(), 0.0) / nq;
        double avg_cmps = (double)std::accumulate(dist_cmps.begin(), dist_cmps.end(), 0ULL) / nq;
        double avg_lat  = std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;
        double avg_beam = (double)std::accumulate(beams.begin(), beams.end(), 0ULL) / nq;

        std::sort(latencies.begin(), latencies.end());
        std::sort(beams.begin(), beams.end());
        auto pct = [&](const std::vector<uint32_t>& v, double p) {
            return v[(size_t)(p * (v.size() - 1))];
        };

        std::cout << std::fixed << std::setprecision(4)
                  << "  R@10=" << avg_r10 << "  R@100=" << avg_r100 << std::endl;
        std::cout << std::fixed << std::setprecision(1)
                  << "  AvgCmps=" << avg_cmps
                  << "  AvgLat=" << avg_lat << "us"
                  << "  P95=" << latencies[(size_t)(0.95 * nq)] << "us"
                  << "  P99=" << latencies[(size_t)(0.99 * nq)] << "us"
                  << "  QPS=" << std::setprecision(0) << qps << std::endl;
        std::cout << "  Effective beam (nodes expanded): avg=" << std::setprecision(1)
                  << avg_beam << "  min=" << beams.front()
                  << "  p50=" << pct(beams, 0.50) << "  p90=" << pct(beams, 0.90)
                  << "  p99=" << pct(beams, 0.99) << "  max=" << beams.back()
                  << std::endl;

        // Coarse histogram of effective beam.
        std::cout << "  Beam distribution:" << std::endl;
        uint32_t bmin = beams.front(), bmax = beams.back();
        const int NB = 10;
        double width = std::max(1u, (bmax - bmin)) / (double)NB;
        std::vector<uint32_t> hist(NB, 0);
        for (uint32_t b : beams) {
            int bin = (int)((b - bmin) / width);
            if (bin >= NB) bin = NB - 1;
            hist[bin]++;
        }
        for (int b = 0; b < NB; b++) {
            uint32_t lo = bmin + (uint32_t)(b * width);
            uint32_t hi = bmin + (uint32_t)((b + 1) * width);
            int bars = (int)(50.0 * hist[b] / nq);
            std::cout << "    [" << std::setw(4) << lo << "-" << std::setw(4) << hi << ") "
                      << std::string(bars, '#') << " " << hist[b] << std::endl;
        }

        std::cout << "\nDone." << std::endl;
        return 0;
    }

    // --- Run search for each L value ---
    std::cout << "\n=== Search Results ===" << std::endl;
    std::cout << std::setw(7) << "L"
              << std::setw(12) << "R@10"
              << std::setw(12) << "R@100"
              << std::setw(14) << "AvgCmps"
              << std::setw(13) << "AvgLat(us)"
              << std::setw(13) << "P95(us)"
              << std::setw(13) << "P99(us)"
              << std::setw(12) << "QPS"
              << std::endl;
    std::cout << std::string(96, '-') << std::endl;

    for (uint32_t L : L_values) {
        std::vector<double> recall10(nq), recall100(nq);
        std::vector<uint32_t> dist_cmps(nq);
        std::vector<double> latencies(nq);

        // Return up to min(L,100) neighbors so the beam width L is preserved
        // (search() would otherwise bump L up to K).
        const uint32_t Kret = std::min(L, R100);

        Timer wall;  // wall-clock over the whole query set => throughput (QPS)
        #pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t q = 0; q < nq; q++) {
            SearchResult res = routing ? index.search_routed(queries.row(q), Kret, L)
                                       : index.search(queries.row(q), Kret, L);

            recall10[q]  = compute_recall(res.ids, gt.row(q), R10);
            recall100[q] = compute_recall(res.ids, gt.row(q), R100);
            dist_cmps[q] = res.dist_cmps;
            latencies[q] = res.latency_us;
        }
        double qps = nq / wall.elapsed_seconds();

        double avg_r10  = std::accumulate(recall10.begin(),  recall10.end(),  0.0) / nq;
        double avg_r100 = std::accumulate(recall100.begin(), recall100.end(), 0.0) / nq;
        double avg_cmps = (double)std::accumulate(dist_cmps.begin(), dist_cmps.end(), 0ULL) / nq;
        double avg_lat  = std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;

        std::sort(latencies.begin(), latencies.end());
        double p95_lat = latencies[(size_t)(0.95 * nq)];
        double p99_lat = latencies[(size_t)(0.99 * nq)];

        std::cout << std::setw(7) << L
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_r10
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_r100
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_cmps
                  << std::setw(13) << std::fixed << std::setprecision(1) << avg_lat
                  << std::setw(13) << std::fixed << std::setprecision(1) << p95_lat
                  << std::setw(13) << std::fixed << std::setprecision(1) << p99_lat
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps
                  << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}
