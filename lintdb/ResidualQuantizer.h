#ifndef LINTDB_RESIDUAL_QUANTIZER_H
#define LINTDB_RESIDUAL_QUANTIZER_H

#include "lintdb/schema/schema_generated.h"
#include "lintdb/api.h"
#include "lintdb/quantizer.h"
#include "lintdb/EmbeddingBlock.h"


namespace lintdb {
    struct ResidualQuantizer: public Quantizer {
        size_t d; // dimensions of the incoming embeddings.
        std::vector<size_t> nbits; // size of the output codes.
        faiss::IndexResidualQuantizer quantizer;

        ResidualQuantizer(size_t d, const std::vector<size_t>& nbits);

        void train(size_t n, const float* x) override;
        // encode will compute the codes for a block of token embeddings.
        void encode(idx_t n, std::vector<float>& data, std::vector<uint8_t>& codes) const override;
        void decode(InvertedDocument* block, float* x) const override;
        void assign(std::vector<EmbeddingBlock> blocks, std::vector<idx_t> coarse_idx) const override;

        bool is_trained() const override;
        size_t n_total() const override;
        size_t code_size() const override;
        faiss::Index* get_inner_quantizer() override;
        size_t dimensions() const override;
    };
}

#endif