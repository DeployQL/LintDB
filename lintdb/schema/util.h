#ifndef LINTDB_SCHEMA_UTIL_H
#define LINTDB_SCHEMA_UTIL_H

#include "lintdb/schema/inverted_index_generated.h"
#include "lintdb/schema/forward_index_generated.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/api.h"

namespace lintdb {
    typedef std::unique_ptr<flatbuffers::FlatBufferBuilder> InvertedDocumentPtr;
    // create an InvertedDocument from an EmbeddingBlock.
    // Flatbuffers should remain an implementation detail of inverted lists.
    std::unique_ptr<flatbuffers::FlatBufferBuilder> create_inverted_index_document(
        const code_t* codes,
        const size_t codes_size
        );

        std::unique_ptr<flatbuffers::FlatBufferBuilder> create_forward_index_document(
        const size_t num_tokens,
        const std::string doc_id,
        const residual_t* residuals,
        const size_t residuals_size
        );
}

#endif