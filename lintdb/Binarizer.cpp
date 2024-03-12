#include "lintdb/Binarizer.h"
#include <algorithm>

namespace lintdb {
    void Binarizer::train(size_t n, const float* x, size_t dim) {
        std::vector<float> avg_residual(dim, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                avg_residual[j] += *(x + i * dim + j);
            }
        }
        for (size_t j = 0; j < dim; ++j) {
            avg_residual[j] /= n;
        }

        calculate_quantiles(avg_residual);

        float total_avg = 0;
        for (size_t i = 0; i < dim; i++) {
            total_avg += avg_residual[i];
        }
        total_avg /= dim;

        this->avg_residual = total_avg;
    }

    void Binarizer::calculate_quantiles(const std::vector<float>& heldout_avg_residual) {
        // Calculate average residual and print it
        float sum = 0.0f;
        for (float value : heldout_avg_residual) {
            sum += std::abs(value);
        }
        avg_residual = sum / heldout_avg_residual.size();

        // Calculate quantiles
        int num_options = 1 << nbits;
        std::vector<float> quantiles;
        for (int i = 0; i < num_options; ++i) {
            quantiles.push_back(static_cast<float>(i) / num_options);
        }

        // Calculate bucket cutoffs and weights
        std::vector<float> bucket_cutoffs_quantiles(quantiles.begin() + 1, quantiles.end());
        std::vector<float> bucket_weights_quantiles;
        for (float quantile : quantiles) {
            bucket_weights_quantiles.push_back(quantile + (0.5f / num_options));
        }

        // Quantile function (assuming sorted data)
        auto quantile_func = [&](float quantile) {
            int index = quantile * heldout_avg_residual.size();
            return heldout_avg_residual[index];
        };

        std::transform(bucket_cutoffs_quantiles.begin(), bucket_cutoffs_quantiles.end(), std::back_inserter(bucket_cutoffs),
                    [&](float quantile) { return quantile_func(quantile); });

        std::transform(bucket_weights_quantiles.begin(), bucket_weights_quantiles.end(), std::back_inserter(bucket_weights),
                    [&](float quantile) { return quantile_func(quantile); });

        }

    std::vector<idx_t> Binarizer::bucketize(const std::vector<float>& residuals) {

    }

    void Binarizer::sa_encode(size_t n, const float* x, residual_t* codes) {

    }

    void Binarizer::sa_decode(size_t n, const residual_t* codes, float* x) {

    }
}