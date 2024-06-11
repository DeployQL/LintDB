#pragma once

#include <vector>
#include <stddef.h>
#include <algorithm>
#include "lintdb/api.h"
#include <limits>

namespace lintdb {
    size_t filter_query_scores(const std::vector<float>& centroid_scores, const size_t num_centroids, const size_t token_position, const float threshold, std::vector<size_t>& sorted_indexes, size_t offset) {
    size_t idx = offset;

    for (size_t j = 0; j < num_centroids; j += 8) {
        for (size_t bit = 0; bit < 8; bit++) {
            size_t index = token_position * num_centroids + j + bit;
            if (index >= centroid_scores.size()) {
                break; // Avoid accessing out-of-bounds memory
            }
            if (centroid_scores[index] >= threshold) {
                sorted_indexes[idx] = j + bit;
                idx++;
            }
        }
    }

    return idx;
}

int popcount(uint32_t t) {
    int count = 0;
    while (t) {
        t &= t - 1;
        count++;
    }
    return count;
}

    inline std::vector<float> compute_ip_with_centroids(
        const std::vector<code_t> &doc_codes, 
        const std::vector<float> &distances, // (num_tokens x num_centroids) 
        const size_t num_centroids,
        const size_t num_tokens
    ) {
        // calculate the scores for each token into a (num_centroids x num_tokens) matrix
        std::vector<float> centroid_scores(num_tokens * doc_codes.size());
        for (size_t i = 0; i < doc_codes.size(); i++) {
            for (size_t j = 0; j < num_tokens; j++) {
                // note the transposition. we create a (num_centroids x num_tokens) matrix
                centroid_scores[i * num_tokens + j] = distances[j * num_centroids + doc_codes[i]];
            }
        }

        return centroid_scores;
    }

    inline float compute_score_by_column_reduction(const std::vector<float>& centroid_distances, const size_t doclen, const size_t num_query_tokens) {
    size_t num_values = centroid_distances.size() / num_query_tokens;

    std::vector<float> max_values(num_query_tokens, std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < num_values; i++) {
        for (size_t j = 0; j < num_query_tokens; j++) {
            float current_value = centroid_distances[i * num_query_tokens + j];
            max_values[j] = std::max(max_values[j], current_value);
        }
    }

    float sum = 0.0f;
    for (float value : max_values) {
        sum += value;
    }

    return sum;
}

inline std::vector<int> filter_centroids_in_scoring(const float th, const float* current_centroid_scores, const size_t doclen) {
    std::vector<int> filtered_indexes;

    for (size_t j = 0; j < doclen; j ++) {
         if (current_centroid_scores[j] > th) {
            filtered_indexes.push_back(static_cast<int>(j));
        }
    }

    return filtered_indexes;
}
}