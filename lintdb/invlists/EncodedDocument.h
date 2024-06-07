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
 * InvertedData holds the data we want to store in the inverted index.
 *
 * If we have N posting lists, we need to return N data structures, one
 * for each posting list.
 */
struct InvertedData{
    code_t key; /// the posting list we're assigning data to.
    idx_t token_id; /// the token id.
    std::string value; /// the serialized data.
};

/**
 * EncodedDocument is the interface between indexes and the inverted list. The
 * data owned by this struct will eventually be stored.
 *
 * Currently, EncodedDocument represent data from both the inverted index and
 * the forward index. This likely means that EncodedDocument also knows
 * how to serialize different messages.
 */
struct EncodedDocument {
    EncodedDocument(
            const std::vector<code_t> codes,
            const std::vector<residual_t> residuals,
            size_t num_tokens,
            idx_t id,
            size_t cs,
            const std::map<std::string, std::string>& metadata = {});

    EncodedDocument(
            const code_t* codes,
            const size_t codes_size,
            const uint8_t* residuals,
            const size_t residuals_size,
            size_t num_tokens,
            idx_t id,
            size_t cs,
            const std::map<std::string, std::string>& metadata = {});

    std::string serialize_metadata() const;

    std::vector<InvertedData>  serialize_inverted_data() const;

    const std::vector<code_t> codes;
    const std::vector<residual_t> residuals;
    const size_t num_tokens;
    idx_t id;
    const std::map<std::string, std::string> metadata;
    const size_t code_size; // residual size per token;
};

/**
 * PartialDocumentCodes hold only a subset of a document's codes.
 *
 * This is used to deserialize codes per token from the inverted index.
 */
struct PartialDocumentCodes {
    idx_t id;
    std::vector<residual_t> partial_residuals;

    explicit PartialDocumentCodes(idx_t id, std::vector<residual_t>& partial_residuals): id(id), partial_residuals(partial_residuals) {}

    static PartialDocumentCodes deserialize(idx_t id, std::string& data);
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