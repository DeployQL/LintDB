#ifndef LINTDB_COARSEQUANTIZER_H
#define LINTDB_COARSEQUANTIZER_H

#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include "lintdb/quantizers/impl/kmeans.h"
#include "lintdb/quantizers/Quantizer.h"
#include <gsl/span>
#include <memory>
#include "lintdb/version.h"

namespace lintdb {

/**
 * @class CoarseQuantizer
 * @brief This class is used for IVF of vectors.
 *
 */
class CoarseQuantizer {
   public:
    bool is_trained = false; // Is the quantizer trained

    explicit CoarseQuantizer(size_t d);
    void train(const size_t n, const float* x, size_t k, size_t num_iter=10);
    void save(const std::string& path);
    void assign(size_t n, const float* x, idx_t* codes);
    void sa_decode(size_t n, const idx_t* codes, float* x);
    void compute_residual(const float* vec, float* residual, idx_t centroid_id);
    void compute_residual_n(int n, const float* vec, float* residual, idx_t* centroid_ids);
    void reconstruct(idx_t centroid_id, float* embedding);
    void search(size_t num_query_tok, const float* data, size_t k_top_centroids, float* distances, idx_t* coarse_idx);
    void reset();
    void add(int n, float* data);
    size_t code_size();
    size_t num_centroids();
    float* get_xb();
    void serialize(const std::string& filename) const;
    static std::unique_ptr<CoarseQuantizer> deserialize(const std::string& filename, const Version& version);

   private:
    size_t d; // Dimensionality of data points
    size_t k; // Number of centroids
    std::vector<float> centroids; // Stored centroids

    uint8_t find_nearest_centroid_index(gsl::span<const float> vec) const;
};

} // namespace lintdb
#endif // LINTDB_COARSEQUANTIZER_H
