#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/schema/schema.pb.h"

namespace lintdb {
EncodedDocument::EncodedDocument(
        const std::vector<code_t>
                c, // reflects the centroid id for each token vector.
        const std::vector<residual_t>
                r, // reflects the residual vector for each token vector.
        size_t num_tokens,
        idx_t id,
        const std::map<std::string, std::string>& metadata)
        : codes(c), residuals(r), num_tokens(num_tokens), id(id), metadata(metadata) {}

EncodedDocument::EncodedDocument(
        const code_t* c, // reflects the centroid id for each token vector.
        const size_t codes_size,
        const uint8_t* r, // reflects the residual vector for each token vector.
        const size_t residuals_size,
        size_t num_tokens,
        idx_t id,
        const std::map<std::string, std::string>& metadata)
        : codes(c, c + codes_size),
          residuals(r, r + residuals_size),
          num_tokens(num_tokens),
          id(id),
          metadata(metadata) {}

std::string EncodedDocument::serialize_metadata() const {
        doc_schema::Schema schema;

        for (const auto& [key, value] : metadata) {
                doc_schema::Field field;
                field.set_key(key);
                field.set_value(value);
                schema.add_fields()->CopyFrom(field);
                
        }
        std::string output;
        schema.SerializeToString(&output);
        return output;
}

DocumentMetadata::DocumentMetadata(idx_t id, doc_schema::Schema metadata): id(id) {
        for (const auto& field : metadata.fields()) {
                std::string key = field.key();
                std::string value = field.value();
                this->metadata[key] = value;
        }
}
} // namespace lintdb