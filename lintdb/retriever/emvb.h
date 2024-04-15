#pragma once

// #ifdef __AVX__

#include <immintrin.h>
#include <vector>
#include <cstdint>
#include "lintdb/api.h"

namespace lintdb {


    size_t filter_query_scores(const std::vector<float> centroid_scores, const size_t num_centroids, const size_t token_position, const float threshold, std::vector<size_t> sorted_indexes, size_t offset) {
        __m256 broad_th = _mm256_set1_ps(threshold);
        __m256 current_values;
        size_t idx = offset;

        for (size_t j=0; j < num_centroids; j+=8) {
            current_values = _mm256_loadu_ps(&centroid_scores[token_position * num_centroids + j]);
            __m256 cmp_result = _mm256_cmp_ps(current_values, broad_th, _CMP_GE_OS);
            int mask = _mm256_movemask_ps(cmp_result);
            if (mask != 0) {

                for (size_t bit = 0; bit < 16; bit++)
                {
                    if ((mask >> bit) & 1)
                    {
                        sorted_indexes[idx] = j + bit;
                        idx++;
                    }
                }
            }
        }

        return idx;
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
    }

     inline float compute_score_by_column_reduction(
        const std::vector<float> &centroid_distances, // should be (num_centroids x num_query_tokens)
        const size_t doclen, 
        const size_t num_query_tokens) // M is num tokens
    {
        // we need to calculate by column the max values. To do this, we'll  break the matrix into chunks of 8 values by column (centroids)
        size_t num_values = centroid_distances.size() / num_query_tokens;

        __m256 sum = _mm256_setzero_ps();

        for (size_t i = 0; i < num_values; i += 8)
        {
            __m256 current = _mm256_loadu_ps(&centroid_distances[i]);

            for (size_t j = 1; j < doclen; j++)
            {
                __m256 current_j = _mm256_loadu_ps(&centroid_distances[j * num_query_tokens + i]);
                __mmask8 m = _mm256_cmp_ps_mask(current_j, current, _CMP_GT_OS);
                current = _mm256_mask_blend_ps(m, current, current_j);
            }

            sum = _mm256_add_ps(sum, current);
        }

        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);

        alignas(32) float result[8];
        _mm256_store_ps(result, sum);
        return result[0]; // Extract the scalar value
    }

inline std::vector<int> filter_centroids_in_scoring(const float th, const float *current_centroid_scores, const size_t doclen)
{
     const int GLOBAL_INDEXES[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<int> filtered_indexes;
    __m256i ids = _mm256_loadu_si256((const __m256i *)GLOBAL_INDEXES);
    const __m256 SHIFT = _mm256_set1_ps(8.0f);
    __m256 broad_th = _mm256_set1_ps(th);
    __m256 current_values;

    size_t avx_cycle_lenth = (doclen / 8) * 8;

    for (size_t j = 0; j < avx_cycle_lenth; j += 8)
    {
        current_values = _mm256_loadu_ps(&current_centroid_scores[j]);
        __m256 mask = _mm256_cmp_ps(current_values, broad_th, _CMP_GT_OS);
        int mask_int = _mm256_movemask_ps(mask);

        for (int i = 0; i < 8; ++i)
        {
            if (mask_int & (1 << i))
            {
                filtered_indexes.push_back(GLOBAL_INDEXES[i]);
            }
        }

        ids = _mm256_add_ps(ids, SHIFT);
    }

    for (size_t j = avx_cycle_lenth; j < doclen; j++)
    {
        if (current_centroid_scores[j] > th)
        {
            filtered_indexes.push_back(j);
        }
    }

    return filtered_indexes;
}
}

// #endif // __AVX__