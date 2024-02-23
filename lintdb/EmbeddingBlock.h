#ifndef LINTDB_EMBEDDING_BLOCK_H
#define LINTDB_EMBEDDING_BLOCK_H

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include "lintdb/api.h"
#include <gsl/span>

namespace lintdb {
    // EmbeddingBlock is a block of embeddings to search. it represents a list of tokens.
    // each embedding is for one token.

    struct EmbeddingBlock{
        gsl::span<float> embeddings;
        size_t num_tokens;
        size_t dimensions;

        EmbeddingBlock() = default;

        EmbeddingBlock(
            const float* embeddings,
            size_t num_tokens,
            size_t dimensions
        ) : num_tokens(num_tokens), dimensions(dimensions) {
            this->embeddings = gsl::span<float>(const_cast<float*>(embeddings), num_tokens * dimensions);
        }

        // // specific to numpy
        // EmbeddingBlock(
        //     float* embeddings,
        //     int num_tokens,
        //     int dimensions
        // ) : num_tokens(num_tokens), dimensions(dimensions) {
        //     this->embeddings = gsl::span<float>(const_cast<float*>(embeddings), num_tokens * dimensions);
        // }
    };
}

#endif