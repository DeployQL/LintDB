#include "lintdb/quantizers/impl/kmeans.h"
#include <vector>
#include <random>
#include <gsl/span>

namespace lintdb {
std::vector<float> kmeans(const std::vector<float>& data, size_t n, size_t dim, size_t k, Metric metric, int iterations) {
    if (data.size() != n * dim) {
        throw std::invalid_argument("Data size does not match the specified dimensions.");
    }

    std::vector<float> centroids(k * dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);

    // Initialize centroids by randomly selecting data points
    for (size_t i = 0; i < k; ++i) {
        size_t index = dis(gen);
        gsl::span<const float> selected_point(data.data() + index * dim, dim);
        std::copy(selected_point.begin(), selected_point.end(), centroids.begin() + i * dim);
    }

    std::vector<size_t> assignments(n);
    std::vector<float> new_centroids(k * dim, 0.0f);
    std::vector<int> counts(k, 0);

    for (int iter = 0; iter < iterations; ++iter) {
        // Assign points to the closest centroid
        for (size_t i = 0; i < n; ++i) {
            gsl::span<const float> point(data.data() + i * dim, dim);
            float best_metric = (metric == Metric::EUCLIDEAN) ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
            size_t best_cluster = 0;
            for (size_t j = 0; j < k; ++j) {
                gsl::span<const float> centroid(centroids.data() + j * dim, dim);
                float current_metric = (metric == Metric::EUCLIDEAN)
                        ? euclidean_distance(point, centroid)
                        : inner_product(point, centroid);
                if ((metric == Metric::EUCLIDEAN && current_metric < best_metric) ||
                    (metric == Metric::INNER_PRODUCT && current_metric > best_metric)) {
                    best_metric = current_metric;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update centroids
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < n; ++i) {
            size_t cluster = assignments[i];
            gsl::span<const float> point(data.data() + i * dim, dim);
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[cluster * dim + d] += point[d];
            }
            counts[cluster]++;
        }

        for (size_t j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (size_t d = 0; d < dim; ++d) {
                    new_centroids[j * dim + d] /= counts[j];
                }
            } else {
                // Reinitialize empty cluster
                size_t index = dis(gen);
                gsl::span<const float> selected_point(data.data() + index * dim, dim);
                std::copy(selected_point.begin(), selected_point.end(), new_centroids.begin() + j * dim);
            }
        }

        centroids.swap(new_centroids);
    }

    return centroids;
}
}