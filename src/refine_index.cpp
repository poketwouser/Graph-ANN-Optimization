#include "vamana_index.h"
#include "io_utils.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index <index_path> --data <fbin> --workload <fbin>"
              << " --output <refined_index_path>"
              << " [--L <collect_beam=75>] [--sample <n=30000>]"
              << " [--keep-min <m=8>] [--min-count <c=1>]"
              << std::endl;
    std::cerr << "NOTE: --workload must be a HELD-OUT set (e.g. sift_learn), "
                 "not the evaluation queries." << std::endl;
}

int main(int argc, char** argv) {
    std::string index_path, data_path, workload_path, output_path;
    uint32_t L = 75, sample = 30000, keep_min = 8, min_count = 1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--index" && i + 1 < argc)         index_path = argv[++i];
        else if (a == "--data" && i + 1 < argc)     data_path = argv[++i];
        else if (a == "--workload" && i + 1 < argc) workload_path = argv[++i];
        else if (a == "--output" && i + 1 < argc)   output_path = argv[++i];
        else if (a == "--L" && i + 1 < argc)        L = std::atoi(argv[++i]);
        else if (a == "--sample" && i + 1 < argc)   sample = std::atoi(argv[++i]);
        else if (a == "--keep-min" && i + 1 < argc) keep_min = std::atoi(argv[++i]);
        else if (a == "--min-count" && i + 1 < argc) min_count = std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    }

    if (index_path.empty() || data_path.empty() || workload_path.empty() ||
        output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== Edge-Usage Refinement (Feature 4) ===" << std::endl;
    VamanaIndex index;
    index.load(index_path, data_path);

    FloatMatrix wl = load_fbin(workload_path);
    uint32_t n = std::min(sample, wl.npts);
    std::cout << "Workload: " << wl.npts << " vectors, using " << n
              << " (collect beam L=" << L << ")" << std::endl;

    index.init_edge_usage();
    Timer t;
    for (uint32_t q = 0; q < n; q++)
        index.collect_edge_usage(wl.row(q), L);
    std::cout << "Edge-usage collection done in " << t.elapsed_seconds() << " s"
              << std::endl;

    double new_deg = index.refine_by_usage(keep_min, min_count);
    std::cout << "New average out-degree: " << new_deg << std::endl;

    index.save(output_path);
    std::cout << "Done." << std::endl;
    return 0;
}
