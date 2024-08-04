#ifndef LINTDB_COARSEQUANTIZER_H
#define LINTDB_COARSEQUANTIZER_H

#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include "lintdb/quantizers/impl/kmeans.h"
#include "lintdb/quantizers/Quantizer.h"
#include <gsl/span>
#include <memory>
#include "lintdb/version.h"
#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>

namespace lintdb {

    /**
     * ICoarseQuantizer defines the interface for coarse quantization.
     *
     * Yes, this needs work to break apart. For now, it's useful in testing and mocking.
     */
    class ICoarseQuantizer {
    public:
        virtual void train(const size_t n, const float* x, size_t k, size_t num_iter) = 0;
        virtual void save(const std::string& path) = 0;
        virtual void assign(size_t n, const float* x, idx_t* codes) = 0;
        virtual void sa_decode(size_t n, const idx_t* codes, float* x) = 0;
        virtual void compute_residual(const float* vec, float* residual, idx_t centroid_id) = 0;
        virtual void compute_residual_n(int n, const float* vec, float* residual, idx_t* centroid_ids) = 0;
        virtual void reconstruct(idx_t centroid_id, float* embedding) = 0;
        virtual void search(size_t num_query_tok, const float* data, size_t k_top_centroids, float* distances, idx_t* coarse_idx) = 0;
        virtual void reset() = 0;
        virtual void add(int n, float* data) = 0;
        virtual size_t code_size() = 0;
        virtual size_t num_centroids() = 0;
        virtual float* get_xb() = 0;
        virtual void serialize(const std::string& filename) const = 0;
        virtual bool is_trained() const = 0;

        virtual ~ICoarseQuantizer() = default;

    };

/**
 * @class CoarseQuantizer
 * @brief This class is used for IVF of vectors.
 *
 */
class CoarseQuantizer: public ICoarseQuantizer {
   public:
    bool is_trained_ = false; // Is the quantizer trained

    explicit CoarseQuantizer(size_t d);
    CoarseQuantizer(size_t d, const std::vector<float>& centroids, size_t k);

    void train(const size_t n, const float* x, size_t k, size_t num_iter=10) override;
    void save(const std::string& path) override;
    void assign(size_t n, const float* x, idx_t* codes) override;
    void sa_decode(size_t n, const idx_t* codes, float* x) override;
    void compute_residual(const float* vec, float* residual, idx_t centroid_id) override;
    void compute_residual_n(int n, const float* vec, float* residual, idx_t* centroid_ids) override;
    void reconstruct(idx_t centroid_id, float* embedding) override;
    void search(size_t num_query_tok, const float* data, size_t k_top_centroids, float* distances, idx_t* coarse_idx) override;
    void reset() override;
    void add(int n, float* data) override;
    size_t code_size() override;
    size_t num_centroids() override;
    float* get_xb() override;
    void serialize(const std::string& filename) const override;
    static std::unique_ptr<CoarseQuantizer> deserialize(const std::string& filename, const Version& version);

    bool is_trained() const override {
        return is_trained_;
    }

   private:
    size_t d; // Dimensionality of data points
    size_t k; // Number of centroids
    std::vector<float> centroids; // Stored centroids

    uint8_t find_nearest_centroid_index(gsl::span<const float> vec) const;
};

class FaissCoarseQuantizer: public ICoarseQuantizer {
    public:
        bool is_trained_ = false; // Is the quantizer trained

        explicit FaissCoarseQuantizer(size_t d);
        FaissCoarseQuantizer(size_t d, const std::vector<float>& centroids, size_t k);

        void train(const size_t n, const float* x, size_t k, size_t num_iter=10) override;
        void save(const std::string& path) override;
        void assign(size_t n, const float* x, idx_t* codes) override;
        void sa_decode(size_t n, const idx_t* codes, float* x) override;
        void compute_residual(const float* vec, float* residual, idx_t centroid_id) override;
        void compute_residual_n(int n, const float* vec, float* residual, idx_t* centroid_ids) override;
        void reconstruct(idx_t centroid_id, float* embedding) override;
        void search(size_t num_query_tok, const float* data, size_t k_top_centroids, float* distances, idx_t* coarse_idx) override;
        void reset() override;
        void add(int n, float* data) override;
        size_t code_size() override;
        size_t num_centroids() override;
        float* get_xb() override;
        void serialize(const std::string& filename) const override;
        static std::unique_ptr<FaissCoarseQuantizer> deserialize(const std::string& filename, const Version& version);

        bool is_trained() const override {
            return is_trained_;
        }

    private:
        size_t d; // Dimensionality of data points
        size_t k; // Number of centroids
        faiss::IndexFlatIP index;

        uint8_t find_nearest_centroid_index(gsl::span<const float> vec) const;
    };

} // namespace lintdb
#endif // LINTDB_COARSEQUANTIZER_H
