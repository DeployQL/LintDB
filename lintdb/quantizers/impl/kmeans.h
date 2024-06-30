#ifndef LINTDB_KMEANS_H
#define LINTDB_KMEANS_H

#include <gsl/span>
#include <vector>
#include <stddef.h>
#include <cmath>

namespace lintdb {

enum class Metric {
    EUCLIDEAN,
    INNER_PRODUCT
};
// Helper function for Euclidean distance
inline float euclidean_distance(gsl::span<const float> a, gsl::span<const float> b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Helper function for inner product
inline float inner_product(gsl::span<const float> a, gsl::span<const float> b) {
    size_t size = a.size();
    size_t i = 0;
    float result = 0.0f;

    // Use manual loop unrolling for better performance
    for (; i + 4 <= size; i += 4) {
        result += a[i] * b[i];
        result += a[i + 1] * b[i + 1];
        result += a[i + 2] * b[i + 2];
        result += a[i + 3] * b[i + 3];
    }

    // Process remaining elements
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// K-means clustering for a single sub-vector
std::vector<float> kmeans(const std::vector<float>& data, size_t n, size_t dim, size_t k, Metric metric, int iterations = 100);

}

#endif // LINTDB_KMEANS_H
