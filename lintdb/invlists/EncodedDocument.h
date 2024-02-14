#ifndef LINTDB_INVLISTS_ENCODED_DOCUMENT_H
#define LINTDB_INVLISTS_ENCODED_DOCUMENT_H

#include <stddef.h>
#include <string>
#include "lintdb/api.h"
#include <vector>
#include <arrayfire.h>

namespace lintdb {
    /**
     * EncodedDocument is the interface between indexes and the inverted list. The data owned by this struct
     * will eventually be stored.
     * 
     * Currently, EncodedDocument represent data from both the inverted index and the forward index. This makes
     * it easier to pass data down, but returning data has to worry about what data is populated.
    */
    struct EncodedDocument {
        EncodedDocument(
            const std::vector<code_t> codes, //reflects the centroid id for each token vector.
            const std::vector<residual_t> residuals, // reflects the residual vector for each token vector.
            size_t num_tokens,
            idx_t id, 
            std::string doc_id
        );

        EncodedDocument(
            const code_t* codes, //reflects the centroid id for each token vector.
            const size_t codes_size,
            const uint8_t* residuals, // reflects the residual vector for each token vector.
            const size_t residuals_size,
            size_t num_tokens,
            idx_t id, 
            std::string doc_id
        );

        const std::vector<code_t> codes;
        const std::vector<residual_t> residuals;
        const size_t num_tokens; // num_tokens
        idx_t id;
        std::string doc_id;
    };
}

#endif