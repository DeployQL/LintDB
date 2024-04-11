#ifndef LINTDB_SEARCH_OPTIONS_H
#define LINTDB_SEARCH_OPTIONS_H

#include <vector>
#include "lintdb/api.h"
#include <stddef.h>

namespace lintdb {
/**
 * SearchOptions enables custom searching behavior. 
 * 
 * These options expose ways to tradeoff recall and latency at different levels of retrieval.
 * Searching more centroids:
 * - decrease centroid_score_threshold and increase k_top_centroids.
 * - increase n_probe in search()
 * 
 * Decreasing latency:
 * - increase centroid_score_threshold and decrease k_top_centroids.
 * - decrease n_probe in search()
*/
struct SearchOptions {
    idx_t expected_id = -1; /// expects a document id in the return result. prints additional information during execution. useful for debugging.
    float centroid_score_threshold = 0.45; /// the threshold for centroid scores. 
    size_t k_top_centroids = 2; /// the number of top centroids to consider. 
    size_t num_second_pass = 1024; /// the number of second pass candidates to consider. 

    SearchOptions(): expected_id(-1) {};
};
}

#endif