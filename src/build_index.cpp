#include "vamana_index.h"
#include "kmeans.h"
#include "timer.h"

#include <cmath>
#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --data <fbin_path>"
              << " --output <index_path>"
              << " [--R <max_degree=32>]"
              << " [--L <build_search_list=75>]"
              << " [--alpha <rng_alpha=1.2>]"
              << " [--gamma <degree_multiplier=1.5>]"
              << std::endl;
}

int main(int argc, char** argv) {
    // Defaults
    std::string data_path, output_path, clusters_path;
    uint32_t R = 32;
    uint32_t L = 75;
    float alpha = 1.2f;
    float gamma = 1.5f;
    float diversity = 0.0f;    // F5 cluster-diversity strength
    float noise_ratio = 0.0f;  // F6 fraction of R added as random edges

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc)       data_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--R" && i + 1 < argc)      R = std::atoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc)      L = std::atoi(argv[++i]);
        else if (arg == "--alpha" && i + 1 < argc)  alpha = std::atof(argv[++i]);
        else if (arg == "--gamma" && i + 1 < argc)  gamma = std::atof(argv[++i]);
        else if (arg == "--clusters" && i + 1 < argc)  clusters_path = argv[++i];
        else if (arg == "--diversity" && i + 1 < argc) diversity = std::atof(argv[++i]);
        else if (arg == "--noise" && i + 1 < argc)     noise_ratio = std::atof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_path.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== Vamana Index Builder ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  R     = " << R << std::endl;
    std::cout << "  L     = " << L << std::endl;
    std::cout << "  alpha = " << alpha << std::endl;
    std::cout << "  gamma = " << gamma << std::endl;

    std::cout << "  diversity = " << diversity
              << (clusters_path.empty() ? " (no clusters)" : "") << std::endl;
    std::cout << "  noise = " << noise_ratio << std::endl;

    VamanaIndex index;

    // F5: load clusters and enable cluster-diverse pruning before building.
    ClusterData cd;  // must outlive build()
    if (!clusters_path.empty() && diversity > 0.0f) {
        cd = load_clusters(clusters_path);
        index.set_build_diversity(&cd.assignment, diversity);
        std::cout << "  Cluster-diverse pruning ON (K=" << cd.K
                  << ", strength=" << diversity << ")" << std::endl;
    }

    Timer total_timer;
    index.build(data_path, R, L, alpha, gamma);

    // F6: inject random long-range edges after build.
    if (noise_ratio > 0.0f) {
        uint32_t epn = (uint32_t)std::lround(noise_ratio * R);
        std::cout << "  Injecting noise: " << epn << " random edges/node" << std::endl;
        index.inject_noise(epn, 1234);
    }

    double total_time = total_timer.elapsed_seconds();

    std::cout << "\nTotal build time: " << total_time << " seconds" << std::endl;

    index.save(output_path);
    std::cout << "Done." << std::endl;
    return 0;
}
