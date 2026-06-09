#pragma once

#include <cstdint>
#include <vector>
#include <string>

// Offline K-Means clustering used for query routing (Feature 1).
//
// We train centroids (k-means++ init + Lloyd iterations) on a random subsample
// of the data for speed, then assign every point to its nearest centroid. For
// each cluster we also record the medoid — the actual data point closest to the
// centroid — which serves as a graph entry point during routed search.
struct ClusterData {
    uint32_t K    = 0;            // number of clusters
    uint32_t dim  = 0;
    uint32_t npts = 0;
    std::vector<float>    centroids;     // K * dim, row-major
    std::vector<uint32_t> assignment;    // npts: cluster id of each point
    std::vector<uint32_t> medoids;       // K: data-point id nearest each centroid
    std::vector<uint32_t> cluster_size;  // K: number of points per cluster

    const float* centroid(uint32_t c) const {
        return centroids.data() + (size_t)c * dim;
    }
};

// Train k-means and assign all points.
//   K:            cluster count
//   iters:        Lloyd iterations on the training subsample
//   train_sample: number of points to subsample for training (0 => all)
//   seed:         RNG seed
ClusterData train_kmeans(const float* data, uint32_t npts, uint32_t dim,
                         uint32_t K, uint32_t iters, uint32_t train_sample,
                         uint32_t seed);

void        save_clusters(const ClusterData& cd, const std::string& path);
ClusterData load_clusters(const std::string& path);
