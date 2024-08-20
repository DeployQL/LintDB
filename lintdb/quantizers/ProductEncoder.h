#pragma once

#include <memory>
#include "lintdb/quantizers/PQDistanceTables.h"
#include "lintdb/quantizers/Quantizer.h"

namespace faiss {
struct IndexPQ;
}

namespace lintdb {
struct ProductEncoder : public Quantizer{
    std::shared_ptr<faiss::IndexPQ> pq;
    size_t nbits; // number of bits used in binarizing the residuals.
    size_t dim;   // number of dimensions per embedding.
    size_t dsub;  // dimensionality of each subvector;
    size_t ksub;  // number of centroids per subquantizer.
    size_t num_subquantizers;

    /// This table is used to precompute the inner product between the centroids
    /// of the PQ quantizer.
    std::vector<float> precomputed_table;

    ProductEncoder(size_t dim, size_t nbits, size_t num_subquantizers);

    ProductEncoder(const ProductEncoder& other);

    friend void swap(ProductEncoder& lhs, ProductEncoder& rhs);

    ProductEncoder& operator=(ProductEncoder& other) {
        swap(*this, other);
        return *this;
    }

    bool is_trained = false;

    void sa_encode(size_t n, const float* x, residual_t* codes) override;
    void sa_decode(size_t n, const residual_t* codes, float* x) override;
    size_t code_size() override;

    size_t get_nbits() override {
        return nbits;
    }

    // Compute the inner product table for the given embeddings.
    // This currently wraps the underlying faiss PQ index.
    std::unique_ptr<PQDistanceTables> get_distance_tables(
            const float* query_data,
            size_t num_tokens) const;

    void save(const std::string path) override;

    static std::unique_ptr<ProductEncoder> load(
            std::string path,
            QuantizerConfig& config);

    void train(const size_t n, const float* embeddings, const size_t dim)
            override;

    QuantizerType get_type() override;
};
} // namespace lintdb