

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

    std::unique_ptr<flatbuffers::FlatBufferBuilder> create_doc_mapping(
        const idx_t* coarse_idx,
        const size_t idx_size
        ) {

        std::vector<uint32_t> casted_codes;
        std::transform(coarse_idx, coarse_idx+idx_size, std::back_inserter(casted_codes), [](idx_t i) { return static_cast<uint32_t>(i); });

        auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
        // auto codes = builder->CreateVector(flat_codes, num_tokens * code_size);
        auto fb_centroids = builder->CreateVector(casted_codes.data(), casted_codes.size());
        
        auto doc = CreateDocumentClusterMapping(*builder, fb_centroids);
        builder->Finish(doc);

        return builder;
    }
}