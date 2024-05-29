#pragma once

#include <memory>
#include "lintdb/quantizers/Quantizer.h"

namespace faiss {
    struct IndexPQ;
}

namespace lintdb {
    struct ProductEncoder : public Quantizer {
        std::shared_ptr<faiss::IndexPQ> pq;
        size_t nbits; // number of bits used in binarizing the residuals.
        size_t dim; // number of dimensions per embedding.
        
        ProductEncoder(
            size_t dim,
            size_t nbits, 
            size_t num_subquantizers
        );

        bool is_trained = false;

        void sa_encode(size_t n, const float* x, residual_t* codes) override;
        void sa_decode(size_t n, const residual_t* codes, float* x) override;
        size_t code_size() override;

        size_t get_nbits() override {
            return nbits;
        }

        void save(const std::string path) override;

        static std::unique_ptr<ProductEncoder> load(std::string path, QuantizerConfig& config);

        void train(const size_t n, const float* embeddings, const size_t dim) override;

        QuantizerType get_type() override;
         
        private:

        
    };
}