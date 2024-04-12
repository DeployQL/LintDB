#pragma once

// #ifdef __AVX__

#include <immintrin.h>
#include <vector>

namespace lintdb {


    size_t filter_doc(const std::vector<float> centroid_scores, const size_t num_centroids, const size_t token_position, const float threshold, size_t* started_sorted) {
        size_t* sorted_indexes = started_sorted;
        __m256 broad_th = _mm_set1_ps(threshold);
        __m256 current_values;
        size_t idx = 0;

        for (size_t j=0; j < n_centroids; j+=8) {
            current_values = _mm256_loadu_ps(&centroid_scores[token_position * num_centroids + j]);
            __m256 mask = _mm256_cmp_ps(current_values, broad_th, _CMP_GE_OS);
            if (mask != (uint16_t)0)
            {

                for (size_t bit = 0; bit < 16; bit++)
                {
                    if ((mask >> bit) & (uint16_t)1)
                    {
                        sorted_indexes[idx] = j + bit;
                        idx++;
                    }
                }
            }
        }
    }
}

// #endif // __AVX2__