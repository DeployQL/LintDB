#ifndef LINTDB_INVLISTS_ENCODED_DOCUMENT_H
#define LINTDB_INVLISTS_ENCODED_DOCUMENT_H

#include <stddef.h>
#include <string>
#include <vector>
#include <memory>
#include "lintdb/api.h"
#include <map>

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
            idx_t id,
            const std::map<std::string, std::string>& metadata = {}
        );

    EncodedDocument(
            const code_t*
                    codes, // reflects the centroid id for each token vector.
            const size_t codes_size,
            const uint8_t* residuals, // reflects the residual vector for each
                                      // token vector.
            const size_t residuals_size,
            size_t num_tokens,
            idx_t id,
            const std::map<std::string, std::string>& metadata = {}
        );

    std::string serialize_metadata() const;

    const std::vector<code_t> codes;
    const std::vector<residual_t> residuals;
    const size_t num_tokens; // num_tokens
    idx_t id;
    const std::map<std::string, std::string> metadata;
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

/**
 * DocumentMetadata is a struct that holds metadata for a document.
 * 
 * When creating a DocumentMetadata object, the metadata is owned by this object.
*/
struct DocumentMetadata {
    std::map<std::string, std::string> metadata;

    DocumentMetadata() = default;

    DocumentMetadata(const std::map<std::string, std::string>& md)
            : metadata(md) {}

    static std::unique_ptr<DocumentMetadata> deserialize(std::string& metadata);
};



} // namespace lintdb


#endif