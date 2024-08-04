#pragma once

#include <memory>
#include <utility>
#include <vector>
#include "lintdb/assert.h"
#include "lintdb/quantizers/CoarseQuantizer.h"

namespace lintdb {

struct QueryTensor {
    std::vector<float> query;
    size_t num_query_tokens;
};

class KnnNearestCentroids {
   public:
    KnnNearestCentroids() = default;
    void calculate(
            std::vector<float>& query,
            const size_t num_query_tokens,
            const std::shared_ptr<ICoarseQuantizer> quantizer,
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

    /// Returns the top centroid id for the idx-th token.
    inline idx_t get_assigned_centroid(size_t idx) const {
        return coarse_idx[idx * total_centroids_to_calculate];
    }

    inline std::vector<float> get_reordered_distances() const {
        return reordered_distances;
    }

    inline bool is_valid() const {
        // this works because we don't set num_centroids until we have
        // calculated them.
        return num_centroids > 0;
    }

    inline QueryTensor get_query_tensor() const {
        LINTDB_THROW_IF_NOT_MSG(!query.empty(), "query is empty");
        return {query, num_query_tokens};
    }

   private:
    std::vector<float> query;
    size_t num_query_tokens;
    size_t num_centroids;
    size_t total_centroids_to_calculate;
    std::vector<std::pair<float, idx_t>> top_centroids;
    std::vector<float> distances;
    std::vector<idx_t> coarse_idx;
    std::vector<float> reordered_distances; /// distances that match the
                                            /// centroid id position.
};

} // namespace lintdb
