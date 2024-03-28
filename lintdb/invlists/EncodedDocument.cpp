#include "lintdb/invlists/EncodedDocument.h"

namespace lintdb {
EncodedDocument::EncodedDocument(
        const std::vector<code_t>
                c, // reflects the centroid id for each token vector.
        const std::vector<residual_t>
                r, // reflects the residual vector for each token vector.
        size_t num_tokens,
        idx_t id)
        : codes(c),
          residuals(r),
          num_tokens(num_tokens),
          id(id) {}

EncodedDocument::EncodedDocument(
        const code_t* c, // reflects the centroid id for each token vector.
        const size_t codes_size,
        const uint8_t* r, // reflects the residual vector for each token vector.
        const size_t residuals_size,
        size_t num_tokens,
        idx_t id)
        : codes(c, c + codes_size),
          residuals(r, r + residuals_size),
          num_tokens(num_tokens),
          id(id) {}
} // namespace lintdb