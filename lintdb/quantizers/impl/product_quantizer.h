#ifndef LINTDB_PRODUCT_QUANTIZER_H
#define LINTDB_PRODUCT_QUANTIZER_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include "lintdb/quantizers/impl/kmeans.h"

namespace lintdb {

class ProductQuantizer {
   public:
    ProductQuantizer(size_t d, size_t m, size_t k, Metric metric = Metric::EUCLIDEAN)
            : d(d), m(m), k(k), metric(metric), subvector_dim(d / m), codebooks(m * k * subvector_dim), is_trained(false) {
        if (d % m != 0) {
            throw std::invalid_argument("The dimensionality must be divisible by the number of sub-quantizers.");
        }
    }

    void fit(const std::vector<std::vector<float>>& data) {
        if (data.empty() || data[0].size() != d) {
            throw std::invalid_argument("Invalid data dimensions.");
        }

        std::vector<float> flattened_data(data.size() * d);
        for (size_t i = 0; i < data.size(); ++i) {
            std::copy(data[i].begin(), data[i].end(), flattened_data.begin() + i * d);
        }

        // Fit k-means for each subvector
        for (size_t i = 0; i < m; ++i) {
            std::vector<float> subvector_data(data.size() * subvector_dim);
            for (size_t j = 0; j < data.size(); ++j) {
                std::copy(flattened_data.begin() + j * d + i * subvector_dim,
                          flattened_data.begin() + j * d + (i + 1) * subvector_dim,
                          subvector_data.begin() + j * subvector_dim);
            }
            auto centroids = kmeans(subvector_data, data.size(), subvector_dim, k, metric);
            std::copy(centroids.begin(), centroids.end(), codebooks.begin() + i * k * subvector_dim);
        }

        is_trained = true;
    }

    std::vector<uint8_t> quantize(const std::vector<float>& vec) const {
        if (vec.size() != d) {
            throw std::invalid_argument("Vector size does not match dimensionality.");
        }

        std::vector<uint8_t> codes(m);

        // Quantize each subvector
        for (size_t i = 0; i < m; ++i) {
            gsl::span<const float> subvec(vec.data() + i * subvector_dim, subvector_dim);
            codes[i] = find_nearest_codebook_index(subvec, i);
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
            gsl::span<const float> centroid(codebooks.data() + (i * k + codes[i]) * subvector_dim, subvector_dim);
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

    void serialize(const std::string& filename) const {
        std::ofstream ofs(filename, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) {
            throw std::runtime_error("Unable to open file for writing.");
        }

        // Write basic information
        ofs.write(reinterpret_cast<const char*>(&d), sizeof(d));
        ofs.write(reinterpret_cast<const char*>(&m), sizeof(m));
        ofs.write(reinterpret_cast<const char*>(&k), sizeof(k));

        // Write metric
        int metric_value = static_cast<int>(metric);
        ofs.write(reinterpret_cast<const char*>(&metric_value), sizeof(metric_value));

        // Write codebooks
        ofs.write(reinterpret_cast<const char*>(codebooks.data()), codebooks.size() * sizeof(float));

        ofs.close();
    }

    void deserialize(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary | std::ios::in);
        if (!ifs.is_open()) {
            throw std::runtime_error("Unable to open file for reading.");
        }

        // Read basic information
        ifs.read(reinterpret_cast<char*>(&d), sizeof(d));
        ifs.read(reinterpret_cast<char*>(&m), sizeof(m));
        ifs.read(reinterpret_cast<char*>(&k), sizeof(k));
        subvector_dim = d / m;

        // Read metric
        int metric_value;
        ifs.read(reinterpret_cast<char*>(&metric_value), sizeof(metric_value));
        metric = static_cast<Metric>(metric_value);

        // Resize codebooks
        codebooks.resize(m * k * subvector_dim);

        // Read codebooks
        ifs.read(reinterpret_cast<char*>(codebooks.data()), codebooks.size() * sizeof(float));

        ifs.close();
    }

    size_t get_M() const {
            return m;
    }

   private:
    size_t d; // Dimensionality of input vectors
    size_t m; // Number of sub-quantizers
    size_t k; // Number of centroids per sub-quantizer
    size_t subvector_dim; // Dimensionality of each subvector
    Metric metric; // Metric to use
    std::vector<float> codebooks; // Flattened codebooks for each sub-quantizer
    bool is_trained;

    uint8_t find_nearest_codebook_index(gsl::span<const float> subvec, size_t subquantizer_index) const {
        float best_metric = (metric == Metric::EUCLIDEAN) ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
        uint8_t best_index = 0;

        for (size_t i = 0; i < k; ++i) {
            gsl::span<const float> centroid(codebooks.data() + (subquantizer_index * k + i) * subvector_dim, subvector_dim);
            float current_metric = (metric == Metric::EUCLIDEAN)
                    ? euclidean_distance(subvec, centroid)
                    : inner_product(subvec, centroid);
            if ((metric == Metric::EUCLIDEAN && current_metric < best_metric) ||
                (metric == Metric::INNER_PRODUCT && current_metric > best_metric)) {
                best_metric = current_metric;
                best_index = static_cast<uint8_t>(i);
            }
        }

        return best_index;
    }
};


} // namespace lintdb

#endif // LINTDB_PRODUCT_QUANTIZER_H
