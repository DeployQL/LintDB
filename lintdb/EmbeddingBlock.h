#ifndef LINTDB_EMBEDDING_BLOCK_H
#define LINTDB_EMBEDDING_BLOCK_H

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace lintdb {
    // EmbeddingBlock is a block of embeddings to search. it represents a list of tokens.
    // each embedding is for one token.
    struct EmbeddingBlock {
        size_t len;
        std::vector<float> data;
        std::string id;
        std::string doc_id;

        EmbeddingBlock(size_t len, const float* data, std::string id, std::string doc_id)
        : len(len), data(data, data + len), id(std::move(id)), doc_id(std::move(doc_id)) {}

        EmbeddingBlock(size_t len, std::vector<float>&& data, std::string id, std::string doc_id)
        : len(len), data(std::move(data)), id(std::move(id)), doc_id(std::move(doc_id)) {}
    };
}

#endif