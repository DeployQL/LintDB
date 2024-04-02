

#include "lintdb/schema/util.h"

namespace lintdb {
// Note: flatbuffer data is little endian.
std::unique_ptr<flatbuffers::FlatBufferBuilder> create_inverted_index_document(
        const code_t* codes,
        const size_t codes_size) {
    auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
    // auto codes = builder->CreateVector(flat_codes, num_tokens * code_size);
    auto fb_codes = builder->CreateVector(codes, codes_size);

    auto doc = CreateInvertedIndexDocument(*builder, fb_codes);
    builder->Finish(doc);

    return builder;
}

std::unique_ptr<flatbuffers::FlatBufferBuilder> create_forward_index_document(
        const size_t num_tokens,
        const residual_t* residuals,
        const size_t residuals_size) {
    auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
    // auto codes = builder->CreateVector(flat_codes, num_tokens * code_size);
    auto fb_residuals = builder->CreateVector(residuals, residuals_size);

    auto doc = CreateForwardIndexDocument(
            *builder, num_tokens, fb_residuals);
    builder->Finish(doc);

    return builder;
}

std::unique_ptr<flatbuffers::FlatBufferBuilder> create_doc_mapping(
        const idx_t* coarse_idx,
        const size_t idx_size) {

    auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
    // auto codes = builder->CreateVector(flat_codes, num_tokens * code_size);
    auto fb_centroids =
            builder->CreateVector(coarse_idx, idx_size);

    auto doc = CreateDocumentClusterMapping(*builder, fb_centroids);
    builder->Finish(doc);

    return builder;
}
} // namespace lintdb