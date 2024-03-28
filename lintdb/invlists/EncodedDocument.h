#ifndef LINTDB_INVLISTS_ENCODED_DOCUMENT_H
#define LINTDB_INVLISTS_ENCODED_DOCUMENT_H

#include <stddef.h>
#include <string>
#include <vector>
#include "lintdb/api.h"

namespace lintdb {
/**
 * EncodedDocument is the interface between indexes and the inverted list. The
 * data owned by this struct will eventually be stored.
 *
 * Currently, EncodedDocument represent data from both the inverted index and
 * the forward index. This makes it easier to pass data down, but returning data
 * has to worry about what data is populated.
 */
struct EncodedDocument {
    EncodedDocument(
            const std::vector<code_t>
                    codes, // reflects the centroid id for each token vector.
            const std::vector<residual_t>
                    residuals, // reflects the residual vector for each token
                               // vector.
            size_t num_tokens,
            idx_t id);

    EncodedDocument(
            const code_t*
                    codes, // reflects the centroid id for each token vector.
            const size_t codes_size,
            const uint8_t* residuals, // reflects the residual vector for each
                                      // token vector.
            const size_t residuals_size,
            size_t num_tokens,
            idx_t id
        );

    const std::vector<code_t> codes;
    const std::vector<residual_t> residuals;
    const size_t num_tokens; // num_tokens
    idx_t id;
};

struct InvertedDocument {
    idx_t id;
};

struct DocumentCodes {
    const idx_t id;
    const std::vector<code_t> codes;
    const size_t num_tokens;

    DocumentCodes(idx_t id, std::vector<code_t> codes, size_t num_tokens)
            : id(id), codes(codes), num_tokens(num_tokens) {}
    
    DocumentCodes(idx_t id, const code_t* codes, size_t codes_size, size_t num_tokens)
            : id(id), codes(codes, codes + codes_size), num_tokens(num_tokens) {}
};

struct DocumentResiduals {
    const idx_t id;
    const std::vector<residual_t> residuals;
    const size_t num_tokens;

    DocumentResiduals(idx_t id, std::vector<residual_t> residuals, size_t num_tokens)
            : id(id), residuals(residuals), num_tokens(num_tokens) {}

    DocumentResiduals(idx_t id, const residual_t* residuals, size_t residuals_size, size_t num_tokens)
            : id(id), residuals(residuals, residuals + residuals_size), num_tokens(num_tokens) {}
};



} // namespace lintdb

#endif