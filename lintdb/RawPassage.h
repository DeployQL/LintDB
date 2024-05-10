#ifndef LINTDB_RAW_DOCUMENT_H
#define LINTDB_RAW_DOCUMENT_H

#include <vector>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/api.h"
#include <map>

namespace lintdb {
/**
 * RawPassage is a simple struct to hold the raw passage data.
 *
 * This represents a document before it's indexed.

*/
struct RawPassage {
    /// embedding_block contains the document's embeddings.
    /// this is an array, and can be any number of embeddings, but they'll all
    /// be indexed together.
    EmbeddingBlock embedding_block;
    /// id is a unique identifier for the document or passage.
    /// it must be an integer. we enable document ids to be strings that we can
    /// lookup after retrieval.
    idx_t id;

    std::map<std::string, std::string> metadata;

    RawPassage() = default;

    RawPassage(
            const float* block, /// the embeddings for the document.
            int num_tokens,     /// the number of tokens in the document.
            int dim,            /// dimensions of the embeddings.
            int64_t id,
            const std::map<std::string, std::string>& metadata = {})
            : embedding_block(block, num_tokens, dim), id(id), metadata(metadata) {}
};
} // namespace lintdb

#endif