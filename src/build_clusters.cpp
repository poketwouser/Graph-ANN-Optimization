#include "kmeans.h"
#include "io_utils.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --data <fbin_path>"
              << " --output <clusters_path>"
              << " --num-clusters <K>"
              << " [--iters <lloyd_iters=15>]"
              << " [--train-sample <n=200000>]"
              << " [--seed <seed=42>]"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string data_path, output_path;
    uint32_t K = 0, iters = 15, train_sample = 200000, seed = 42;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc)              data_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc)       output_path = argv[++i];
        else if (arg == "--num-clusters" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc)        iters = std::atoi(argv[++i]);
        else if (arg == "--train-sample" && i + 1 < argc) train_sample = std::atoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc)         seed = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    if (data_path.empty() || output_path.empty() || K == 0) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== K-Means Cluster Builder ===" << std::endl;
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    std::cout << "  Points: " << mat.npts << ", Dimensions: " << mat.dims << std::endl;

    Timer t;
    ClusterData cd = train_kmeans(mat.data.get(), mat.npts, mat.dims,
                                  K, iters, train_sample, seed);
    std::cout << "Clustering done in " << t.elapsed_seconds() << " s" << std::endl;

    save_clusters(cd, output_path);
    std::cout << "Done." << std::endl;
    return 0;
}
