#pragma once

#include <vector>
#include <utility>
#include <memory>
#include "lintdb/quantizers/CoarseQuantizer.h"

namespace lintdb {

    class KnnNearestCentroids {
    public:
        KnnNearestCentroids() = default;
        void calculate(
                const std::vector<float>& query,
                const size_t num_query_tokens,
                const std::shared_ptr<CoarseQuantizer> quantizer,
                const size_t total_centroids_to_calculate);

        std::vector<std::pair<float, idx_t>> get_top_centroids(
                const size_t k_top_centroids, /// k centroids per token to consider.
                const size_t n_probe /// overall number of centroids to return.
        ) const;

        inline std::vector<float> get_distances() const {
            return distances;
        }
        inline std::vector<idx_t> get_indices() const {
            return coarse_idx;
        }

        inline bool is_valid() const {
            // this works because we don't set num_centroids until we have calculated them.
            return num_centroids > 0;
        }

    private:
        size_t num_query_tokens;
        size_t num_centroids;
        std::vector<std::pair<float, idx_t>> top_centroids;
        std::vector<float> distances;
        std::vector<idx_t> coarse_idx;
    };

} // lintdb
