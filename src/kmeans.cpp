#include "kmeans.h"
#include "distance.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

// ============================================================================
// k-means++ initialization (on the training subsample)
// ============================================================================
// Pick K initial centers: the first uniformly at random, each subsequent center
// chosen with probability proportional to its squared distance to the nearest
// already-chosen center (D^2 weighting). Gives well-spread seeds and far fewer
// Lloyd iterations than random init.

static void kmeanspp_init(const float* data, const std::vector<uint32_t>& sample,
                          uint32_t dim, uint32_t K, std::mt19937& rng,
                          std::vector<float>& centroids) {
    const uint32_t n = (uint32_t)sample.size();
    centroids.resize((size_t)K * dim);

    // First center: uniform random sample point.
    uint32_t first = sample[rng() % n];
    std::copy_n(data + (size_t)first * dim, dim, centroids.begin());

    std::vector<float> d2(n, std::numeric_limits<float>::max());

    for (uint32_t c = 1; c < K; c++) {
        // Update nearest-center squared distance using the center just added.
        const float* last = centroids.data() + (size_t)(c - 1) * dim;
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (uint32_t i = 0; i < n; i++) {
            float dist = compute_l2sq(data + (size_t)sample[i] * dim, last, dim);
            if (dist < d2[i]) d2[i] = dist;
            sum += d2[i];
        }

        // Sample the next center with probability proportional to d2.
        std::uniform_real_distribution<double> unif(0.0, sum);
        double target = unif(rng);
        double acc = 0.0;
        uint32_t pick = sample[n - 1];
        for (uint32_t i = 0; i < n; i++) {
            acc += d2[i];
            if (acc >= target) { pick = sample[i]; break; }
        }
        std::copy_n(data + (size_t)pick * dim, dim, centroids.begin() + (size_t)c * dim);
    }
}

// ============================================================================
// Train
// ============================================================================

ClusterData train_kmeans(const float* data, uint32_t npts, uint32_t dim,
                         uint32_t K, uint32_t iters, uint32_t train_sample,
                         uint32_t seed) {
    if (K == 0 || K > npts)
        throw std::runtime_error("Invalid cluster count K");

    std::mt19937 rng(seed);

    // --- Build training subsample ---
    uint32_t n_train = (train_sample == 0 || train_sample > npts) ? npts : train_sample;
    std::vector<uint32_t> sample(npts);
    std::iota(sample.begin(), sample.end(), 0u);
    if (n_train < npts) {
        std::shuffle(sample.begin(), sample.end(), rng);
        sample.resize(n_train);
    }

    std::cout << "  k-means: K=" << K << ", train sample=" << n_train
              << ", iters=" << iters << std::endl;

    // --- k-means++ init ---
    std::vector<float> centroids;
    kmeanspp_init(data, sample, dim, K, rng, centroids);

    // --- Lloyd iterations on the subsample ---
    std::vector<uint32_t> assign_train(n_train, 0);
    std::vector<double> sums((size_t)K * dim);
    std::vector<uint32_t> counts(K);

    for (uint32_t it = 0; it < iters; it++) {
        // Assignment step.
        #pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i < n_train; i++) {
            const float* v = data + (size_t)sample[i] * dim;
            float best = std::numeric_limits<float>::max();
            uint32_t bc = 0;
            for (uint32_t c = 0; c < K; c++) {
                float d = compute_l2sq(v, centroids.data() + (size_t)c * dim, dim);
                if (d < best) { best = d; bc = c; }
            }
            assign_train[i] = bc;
        }

        // Update step (means).
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0u);
        for (uint32_t i = 0; i < n_train; i++) {
            uint32_t c = assign_train[i];
            const float* v = data + (size_t)sample[i] * dim;
            double* s = sums.data() + (size_t)c * dim;
            for (uint32_t d = 0; d < dim; d++) s[d] += v[d];
            counts[c]++;
        }
        for (uint32_t c = 0; c < K; c++) {
            if (counts[c] == 0) continue;  // keep empty cluster's old centroid
            float* cen = centroids.data() + (size_t)c * dim;
            double* s = sums.data() + (size_t)c * dim;
            for (uint32_t d = 0; d < dim; d++)
                cen[d] = (float)(s[d] / counts[c]);
        }
    }

    // --- Final assignment of ALL points + medoid tracking ---
    ClusterData cd;
    cd.K = K; cd.dim = dim; cd.npts = npts;
    cd.centroids = std::move(centroids);
    cd.assignment.resize(npts);
    cd.cluster_size.assign(K, 0);
    cd.medoids.assign(K, UINT32_MAX);

    std::vector<float> medoid_dist(K, std::numeric_limits<float>::max());

    #pragma omp parallel
    {
        // Thread-local medoid candidates to avoid contention.
        std::vector<float> local_md(K, std::numeric_limits<float>::max());
        std::vector<uint32_t> local_medoid(K, UINT32_MAX);

        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < npts; i++) {
            const float* v = data + (size_t)i * dim;
            float best = std::numeric_limits<float>::max();
            uint32_t bc = 0;
            for (uint32_t c = 0; c < K; c++) {
                float d = compute_l2sq(v, cd.centroid(c), dim);
                if (d < best) { best = d; bc = c; }
            }
            cd.assignment[i] = bc;
            if (best < local_md[bc]) { local_md[bc] = best; local_medoid[bc] = i; }
        }

        #pragma omp critical
        {
            for (uint32_t c = 0; c < K; c++) {
                if (local_medoid[c] != UINT32_MAX && local_md[c] < medoid_dist[c]) {
                    medoid_dist[c] = local_md[c];
                    cd.medoids[c] = local_medoid[c];
                }
            }
        }
    }

    for (uint32_t i = 0; i < npts; i++) cd.cluster_size[cd.assignment[i]]++;

    // Repair any empty cluster: point its medoid at the global point 0 so the
    // routing code never dereferences UINT32_MAX.
    for (uint32_t c = 0; c < K; c++)
        if (cd.medoids[c] == UINT32_MAX) cd.medoids[c] = 0;

    // Report balance.
    uint32_t mn = npts, mx = 0;
    for (uint32_t c = 0; c < K; c++) {
        mn = std::min(mn, cd.cluster_size[c]);
        mx = std::max(mx, cd.cluster_size[c]);
    }
    std::cout << "  cluster sizes: min=" << mn << " max=" << mx
              << " avg=" << (npts / K) << std::endl;

    return cd;
}

// ============================================================================
// Persistence
// ============================================================================
// Format: [K][dim][npts] then centroids(K*dim f32), assignment(npts u32),
//         medoids(K u32), cluster_size(K u32).

void save_clusters(const ClusterData& cd, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open clusters file for writing: " + path);
    out.write(reinterpret_cast<const char*>(&cd.K), 4);
    out.write(reinterpret_cast<const char*>(&cd.dim), 4);
    out.write(reinterpret_cast<const char*>(&cd.npts), 4);
    out.write(reinterpret_cast<const char*>(cd.centroids.data()),
              (std::streamsize)cd.centroids.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(cd.assignment.data()),
              (std::streamsize)cd.assignment.size() * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(cd.medoids.data()),
              (std::streamsize)cd.medoids.size() * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(cd.cluster_size.data()),
              (std::streamsize)cd.cluster_size.size() * sizeof(uint32_t));
    std::cout << "Clusters saved to " << path << std::endl;
}

ClusterData load_clusters(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open clusters file: " + path);
    ClusterData cd;
    in.read(reinterpret_cast<char*>(&cd.K), 4);
    in.read(reinterpret_cast<char*>(&cd.dim), 4);
    in.read(reinterpret_cast<char*>(&cd.npts), 4);
    cd.centroids.resize((size_t)cd.K * cd.dim);
    cd.assignment.resize(cd.npts);
    cd.medoids.resize(cd.K);
    cd.cluster_size.resize(cd.K);
    in.read(reinterpret_cast<char*>(cd.centroids.data()),
            (std::streamsize)cd.centroids.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(cd.assignment.data()),
            (std::streamsize)cd.assignment.size() * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(cd.medoids.data()),
            (std::streamsize)cd.medoids.size() * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(cd.cluster_size.data()),
            (std::streamsize)cd.cluster_size.size() * sizeof(uint32_t));
    if (!in.good())
        throw std::runtime_error("Failed to read clusters file: " + path);
    std::cout << "Clusters loaded: K=" << cd.K << ", npts=" << cd.npts << std::endl;
    return cd;
}
