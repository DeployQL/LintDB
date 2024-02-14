

#include "lintdb/schema/util.h"

namespace lintdb {
    // Note: flatbuffer data is little endian.
    std::unique_ptr<flatbuffers::FlatBufferBuilder> create_inverted_index_document(
        const code_t* codes,
        const size_t codes_size
        ) {

        auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
        // auto codes = builder->CreateVector(flat_codes, num_tokens * code_size);
        auto fb_codes = builder->CreateVector(codes, codes_size);
        
        auto doc = CreateInvertedIndexDocument(*builder, fb_codes);
        builder->Finish(doc);

        return builder;
    }

    std::unique_ptr<flatbuffers::FlatBufferBuilder> create_forward_index_document(
        const size_t num_tokens,
        const std::string doc_id,
        const residual_t* residuals,
        const size_t residuals_size
        ) {

        auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
        // auto codes = builder->CreateVector(flat_codes, num_tokens * code_size);
        auto fb_residuals = builder->CreateVector(residuals, residuals_size);
        auto fb_doc_id = builder->CreateString(doc_id);
        
        auto doc = CreateForwardIndexDocument(*builder, fb_doc_id, num_tokens, fb_residuals);
        builder->Finish(doc);

        return builder;
    }
}