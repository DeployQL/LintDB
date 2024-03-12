#ifndef LINTDB_BINARIZER_H
#define LINTDB_BINARIZER_H

#include <vector>
#include <stddef.h>
#include "lintdb/api.h"

namespace lintdb {
    struct Binarizer {
        std::vector<float> bucket_cutoffs;
        std::vector<float> bucket_weights;
        float avg_residual;
        size_t nbits;

        Binarizer(size_t nbits): nbits(nbits) {}

        void train(size_t n, const float* x, size_t dim);
        void calculate_quantiles(const std::vector<float>& heldoout_residual);

        void sa_encode(size_t n, const float* x, residual_t* codes);
        void sa_decode(size_t n, const residual_t* codes, float* x);
        std::vector<idx_t> bucketize(const std::vector<float>& residuals);

    };
}

#endif