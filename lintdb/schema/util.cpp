

#include "lintdb/schema/util.h"

namespace lintdb {
    std::unique_ptr<flatbuffers::FlatBufferBuilder> create_inverted_document(
        const size_t num_tokens,
        const idx_t id,
        const std::string doc_id,
        const uint8_t* flat_codes, 
        size_t code_size
        ) {

        auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
        auto codes = builder->CreateVector(flat_codes, num_tokens * code_size);
        auto fb_doc_id = builder->CreateString(doc_id);
        
        auto doc = CreateInvertedDocument(*builder, fb_doc_id, num_tokens, codes);

        auto dd = GetInvertedDocument(builder->GetBufferPointer());

        return builder;
    }
}