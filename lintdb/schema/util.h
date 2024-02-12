#ifndef LINTDB_SCHEMA_UTIL_H
#define LINTDB_SCHEMA_UTIL_H

#include "lintdb/schema/schema_generated.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/api.h"

namespace lintdb {
    typedef std::unique_ptr<flatbuffers::FlatBufferBuilder> InvertedDocumentPtr;
    // create an InvertedDocument from an EmbeddingBlock.
    std::unique_ptr<flatbuffers::FlatBufferBuilder> create_inverted_document(
        const size_t num_tokens,
        const idx_t id,
        const std::string doc_id,
        const uint8_t* flat_codes, 
        size_t code_size
        );
}

#endif