#ifndef LINTDB_SEARCH_OPTIONS_H
#define LINTDB_SEARCH_OPTIONS_H

#include <cstddef>
#include <vector>
#include <string>
#include "lintdb/api.h"

namespace lintdb {
enum class IndexEncoding {
    NONE,
    BINARIZER,
    PRODUCT_QUANTIZER,
    XTR,
};

/**
 * SearchOptions enables custom searching behavior.
 *
 * These options expose ways to tradeoff recall and latency at different levels
 * of retrieval. Searching more centroids:
 * - decrease centroid_score_threshold and increase k_top_centroids.
 * - increase n_probe in search()
 *
 * Decreasing latency:
 * - increase centroid_score_threshold and decrease k_top_centroids.
 * - decrease n_probe in search()
 */
struct SearchOptions {
    idx_t expected_id = -1; /// expects a document id in the return result.
                            /// prints additional information during execution.
                            /// useful for debugging.
    float centroid_score_threshold =
            0.45; /// the threshold for centroid scores.
    size_t k_top_centroids =
            2; /// the number of top centroids to consider per token.
    size_t num_second_pass =
            1024;        /// the number of second pass candidates to consider.
    size_t n_probe = 32; /// the number of centroids to search overall.
    size_t nearest_tokens_to_fetch =
            100; /// the number of nearest tokens to fetch in XTR.
    std::string colbert_field = "colbert";

    SearchOptions() : expected_id(-1){};
};
} // namespace lintdb

#endif