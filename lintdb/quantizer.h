#ifndef LINTDB_QUANTIZER_H
#define LINTDB_QUANTIZER_H

#include "faiss/Index.h"
#include <faiss/IVFlib.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/schema/schema_generated.h"
#include "lintdb/api.h"

namespace lintdb {
    struct Quantizer {
        virtual bool is_trained() const = 0;
        virtual size_t n_total() const = 0;
        virtual size_t code_size() const = 0;
        virtual faiss::Index* get_inner_quantizer() = 0;
        virtual size_t dimensions() const = 0;
        /**
         * Train the quantizer.
         * 
         * This uses per-token embeddings.
        */
        virtual void train(size_t n, const float* x) = 0;

        /**
         * Compute codes for a block of token embeddings.
         * 
         * @param x the input vectors, size (num_tokens * n) * d
        */
        virtual void encode(idx_t n,std::vector<float>& data, std::vector<uint8_t>& codes) const = 0;

        virtual void decode(InvertedDocument* block, float* x) const = 0;

        // assign is a helper method to assign a block of embeddings to their nearest centroids.
        // coarse_idx should point from each individual token embedding to its centroid.
        // therefore, we have block->len * n size of coarse_idx.
        virtual void assign(std::vector<EmbeddingBlock> blocks, std::vector<idx_t> coarse_idx) const = 0;
    };
}

#endif