#ifndef LINTDB_EMBEDDING_BLOCK_H
#define LINTDB_EMBEDDING_BLOCK_H

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <arrayfire.h>

namespace lintdb {
    // EmbeddingBlock is a block of embeddings to search. it represents a list of tokens.
    // each embedding is for one token.
    typedef af::array EmbeddingBlock;

    const size_t TOKEN_DIMENSION = 0;
    const size_t EMBEDDING_DIMENSION = 1;
}

#endif