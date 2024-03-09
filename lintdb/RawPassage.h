#ifndef LINTDB_RAW_DOCUMENT_H
#define LINTDB_RAW_DOCUMENT_H

#include <vector>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/api.h"

namespace lintdb {
struct RawPassage {
    // embedding_block contains the document's embeddings.
    // this is an array, and can be any number of embeddings, but they'll all be
    // indexed together.
    EmbeddingBlock embedding_block;
    // id is a unique identifier for the document or passage.
    // it must be an integer. we enable document ids to be strings that we can
    // lookup after retrieval.
    idx_t id;
    // document id is a string uuid for the passage.
    std::string doc_id;
    std::string text;

    RawPassage() = default;

    RawPassage(
            const float* block,
            int num_tokens,
            int dim,
            int64_t id,
            std::string doc_id,
            std::string text)
            : id(id), doc_id(doc_id), embedding_block(block, num_tokens, dim), text(text) {}
};
} // namespace lintdb

#endif