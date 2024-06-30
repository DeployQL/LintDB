#pragma once

#include <memory>
#include "lintdb/Encoder.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/Document.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/schema/schema.h"

namespace lintdb {
class CoarseQuantizer;

class EncodedDocument {
   public:
    std::unordered_map<std::string, std::string> encoded_fields;
    std::unordered_map<std::string, std::vector<int>> ivf_centroids;


    void addField(const std::string& name, const std::string& value) {
        encoded_fields[name] = value;
    }

    void addCentroids(const std::string& name, const std::vector<idx_t>& centroids) {
        ivf_centroids[name] = centroids;
    }
};

class DocumentProcessor {
   private:
    Schema schema;
    std::unordered_map<std::string, Field> field_map;
    // each tensor/tensor_array field has a quantizer
    std::unordered_map<std::string, std::shared_ptr<Quantizer>> quantizer_map;
    std::unordered_map<std::string, std::shared_ptr<CoarseQuantizer>> coarse_quantizer_map;

   public:
    DocumentProcessor(
        const Schema& schema,
        const std::shared_ptr<Encoder> encoder,
        const std::shared_ptr<Quantizer> quantizer
    );
    EncodedDocument processDocument(const Document& document);

   private:
    void validateField(const Field& field, const FieldValue& value);
    std::string encodeField(const Field& field, const FieldValue& value);
    void serializeEncodedDocument(
        const EncodedDocument& encodedDoc,
        InvertedIndexDocument& invertedDoc,
        ForwardIndexDocument& forwardDoc
    );
    FieldValue quantizeField(const Field& field, const FieldValue& value);
    std::vector<idx_t> DocumentProcessor::assignIVFCentroids(const Field& field, const FieldValue& value)
};

} // namespace lintdb

