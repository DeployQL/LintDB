#ifndef PRODUCT_QUANTIZER_H
#define PRODUCT_QUANTIZER_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <iostream>

namespace lintdb {
namespace quantizers {

enum class Metric {
    EUCLIDEAN,
    INNER_PRODUCT
};

// A helper function for Euclidean distance
inline float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// A helper function for inner product
inline float inner_product(const std::vector<float>& a, const std::vector<float>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

// K-means clustering for a single sub-vector
std::vector<std::vector<float>> kmeans(const std::vector<std::vector<float>>& data, int k, Metric metric, int iterations = 100) {
    std::vector<std::vector<float>> centroids(k);
    size_t dim = data[0].size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    // Initialize centroids by randomly selecting data points
    for (int i = 0; i < k; ++i) {
        centroids[i] = data[dis(gen)];
    }

    std::vector<int> assignments(data.size());
    for (int iter = 0; iter < iterations; ++iter) {
        // Assign points to the closest centroid
        for (size_t i = 0; i < data.size(); ++i) {
            float best_metric = (metric == Metric::EUCLIDEAN) ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
            int best_cluster = -1;
            for (int j = 0; j < k; ++j) {
                float current_metric = (metric == Metric::EUCLIDEAN)
                                       ? euclidean_distance(data[i], centroids[j])
                                       : inner_product(data[i], centroids[j]);
                if ((metric == Metric::EUCLIDEAN && current_metric < best_metric) ||
                    (metric == Metric::INNER_PRODUCT && current_metric > best_metric)) {
                    best_metric = current_metric;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update centroids
        std::vector<std::vector<float>> new_centroids(k, std::vector<float>(dim, 0.0f));
        std::vector<int> counts(k, 0);
        for (size_t i = 0; i < data.size(); ++i) {
            int cluster = assignments[i];
            std::transform(data[i].begin(), data[i].end(), new_centroids[cluster].begin(),
                           new_centroids[cluster].begin(), std::plus<float>());
            counts[cluster]++;
        }
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                std::transform(new_centroids[j].begin(), new_centroids[j].end(), new_centroids[j].begin(),
                               [=](float val) { return val / counts[j]; });
            } else {
                new_centroids[j] = data[dis(gen)];
            }
        }
        centroids = std::move(new_centroids);
    }
    return centroids;
}

class ProductQuantizer {
public:
    ProductQuantizer(size_t d, size_t m, size_t k, Metric metric = Metric::EUCLIDEAN)
        : d(d), m(m), k(k), metric(metric), subvector_dim(d / m), codebooks(m, std::vector<std::vector<float>>(k, std::vector<float>(subvector_dim))) {
        if (d % m != 0) {
            throw std::invalid_argument("The dimensionality must be divisible by the number of sub-quantizers.");
        }
    }

    void fit(const std::vector<std::vector<float>>& data) {
        if (data.empty() || data[0].size() != d) {
            throw std::invalid_argument("Invalid data dimensions.");
        }

        std::vector<std::vector<std::vector<float>>> subvector_data(m);

        // Split data into subvectors
        for (const auto& vec : data) {
            for (size_t i = 0; i < m; ++i) {
                subvector_data[i].emplace_back(vec.begin() + i * subvector_dim, vec.begin() + (i + 1) * subvector_dim);
            }
        }

        // Fit k-means for each subvector
        for (size_t i = 0; i < m; ++i) {
            codebooks[i] = kmeans(subvector_data[i], k, metric);
        }
    }

    std::vector<uint8_t> quantize(const std::vector<float>& vec) const {
        if (vec.size() != d) {
            throw std::invalid_argument("Vector size does not match dimensionality.");
        }

        std::vector<uint8_t> codes(m);

        // Quantize each subvector
        for (size_t i = 0; i < m; ++i) {
            std::vector<float> subvec(vec.begin() + i * subvector_dim, vec.begin() + (i + 1) * subvector_dim);
            codes[i] = find_nearest_codebook_index(subvec, codebooks[i]);
        }

        return codes;
    }

    std::vector<float> decode(const std::vector<uint8_t>& codes) const {
        if (codes.size() != m) {
            throw std::invalid_argument("Codes size does not match number of sub-quantizers.");
        }

        std::vector<float> vec(d);

        // Decode each subvector
        for (size_t i = 0; i < m; ++i) {
            const auto& centroid = codebooks[i][codes[i]];
            std::copy(centroid.begin(), centroid.end(), vec.begin() + i * subvector_dim);
        }

        return vec;
    }

    std::vector<std::vector<uint8_t>> compute_codes(const std::vector<std::vector<float>>& data) const {
        std::vector<std::vector<uint8_t>> codes;
        codes.reserve(data.size());

        for (const auto& vec : data) {
            codes.push_back(quantize(vec));
        }

        return codes;
    }

private:
    size_t d; // Dimensionality of input vectors
    size_t m; // Number of sub-quantizers
    size_t k; // Number of centroids per sub-quantizer
    size_t subvector_dim; // Dimensionality of each subvector
    Metric metric; // Metric to use

    std::vector<std::vector<std::vector<float>>> codebooks; // Codebooks for each sub-quantizer

    uint8_t find_nearest_codebook_index(const std::vector<float>& subvec, const std::vector<std::vector<float>>& codebook) const {
        float best_metric = (metric == Metric::EUCLIDEAN) ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
        uint8_t best_index = 0;

        for (size_t i = 0; i < codebook.size(); ++i) {
            float current_metric = (metric == Metric::EUCLIDEAN)
                                   ? euclidean_distance(subvec, codebook[i])
                                   : inner_product(subvec, codebook[i]);
            if ((metric == Metric::EUCLIDEAN && current_metric < best_metric) ||
                (metric == Metric::INNER_PRODUCT && current_metric > best_metric)) {
                best_metric = current_metric;
                best_index = static_cast<uint8_t>(i);
            }
        }

        return best_index;
    }
};

// Example usage
int main() {
    size_t d = 4; // Dimensionality
    size_t m = 2; // Number of sub-quantizers
    size_t k = 2; // Number of centroids per sub-quantizer
    Metric metric = Metric::INNER_PRODUCT; // Metric to use

    ProductQuantizer pq(d, m, k, metric);

    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {1.1f, 2.1f, 3.1f, 4.1f},
        {2.0f, 3.0f, 4.0f, 5.0f},
        {2.1f, 3.1f, 4.1f, 5.1f}
    };

    pq.fit(data);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};



} // quantizers
} // lintdb

#endif //PRODUCT_QUANTIZER_H
