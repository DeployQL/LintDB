#include "lintdb/ResidualQuantizer.h"
#include <iostream>

namespace lintdb {
    ResidualQuantizer::ResidualQuantizer (
        size_t d,
        const std::vector<size_t>& nbits
    ): quantizer(d, nbits), d(d), nbits(nbits){}

    void ResidualQuantizer::train(size_t n, const float* x) {
        quantizer.train(n, x);
    }

    void ResidualQuantizer::encode(idx_t n_data, std::vector<float>& data, std::vector<uint8_t>& codes) const {
        quantizer.sa_encode(n_data, data.data(), codes.data());
    }

    void ResidualQuantizer::decode(InvertedDocument* inverted_block, float* x) const {
        // we have num_tokens * n embeddings. each embedding is d dimensions.
        // we want to assign the original embeddings to x based on the code we have.
        quantizer.sa_decode(inverted_block->num_tokens(), inverted_block->codes()->data(), x);
    }

    void ResidualQuantizer::assign(std::vector<EmbeddingBlock> blocks, std::vector<idx_t> coarse_idx) const {
        // we have num_tokens * n embeddings. each embedding is d dimensions.
        // we want to assign the original embeddings to x based on the code we have.
        size_t ntotal = 0;
        for (size_t i = 0; i < blocks.size(); i++) {
            quantizer.assign(1, blocks[i].data.data(), coarse_idx.data() + ntotal);
            ntotal += blocks[i].len;
        }
    }

    bool ResidualQuantizer::is_trained() const {
        return quantizer.is_trained;
    };

    size_t ResidualQuantizer::n_total() const {
        return quantizer.ntotal;
    };

    size_t ResidualQuantizer::code_size() const {
        return quantizer.sa_code_size();
    };

    size_t ResidualQuantizer::dimensions() const {
        return d;
    };

    faiss::Index* ResidualQuantizer::get_inner_quantizer() {
        return &quantizer;
    };

}