#pragma once

#include <memory>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/IndexPQ.h>
#include "lintdb/quantizers/Quantizer.h"

namespace lintdb {
    struct ProductEncoder : public Quantizer {
        ProductEncoder(
            size_t dim,
            size_t nbits, 
            size_t num_subquantizers
        );

        bool is_trained = false;

        void sa_encode(size_t n, const float* x, residual_t* codes) override;
        void sa_decode(size_t n, const residual_t* codes, float* x) override;

        void save(const std::string path) override;

        static std::unique_ptr<ProductEncoder> load(std::string path, QuantizerConfig& config);

        void train(const size_t n, const float* embeddings, const size_t dim) override;

        QuantizerType get_type() override;
         
        private:
        size_t nbits; // number of bits used in binarizing the residuals.
        size_t dim; // number of dimensions per embedding.
        std::unique_ptr<faiss::IndexPQ> pq;
    };
}