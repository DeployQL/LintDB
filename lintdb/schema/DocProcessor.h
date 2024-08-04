#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "lintdb/invlists/IndexWriter.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/Document.h"
#include "lintdb/schema/FieldMapper.h"
#include "lintdb/schema/ProcessedData.h"
#include "lintdb/schema/Schema.h"

namespace lintdb {

class DocumentProcessor {
   public:
    DocumentProcessor(
            const Schema& schema,
            const std::unordered_map<std::string, std::shared_ptr<Quantizer>>&
                    quantizer_map,
            const std::unordered_map<
                    std::string,
                    std::shared_ptr<ICoarseQuantizer>>& coarse_quantizer_map,
            const std::shared_ptr<FieldMapper> field_mapper,
            std::unique_ptr<IIndexWriter> index_writer);
    void processDocument(const uint64_t tenant, const Document& document);

   private:
    static void validateField(const Field& field, const FieldValue& value);
    FieldValue quantizeField(const Field& field, const FieldValue& value);
    std::vector<idx_t> assignIVFCentroids(
            const Field& field,
            const FieldValue& value);

    Schema schema;
    std::unordered_map<std::string, Field> field_map;
    const std::shared_ptr<FieldMapper> field_mapper;
    // each tensor/tensor_array field has a quantizer
    const std::unordered_map<std::string, std::shared_ptr<Quantizer>>&
            quantizer_map;
    const std::unordered_map<std::string, std::shared_ptr<ICoarseQuantizer>>&
            coarse_quantizer_map;

    std::unique_ptr<IIndexWriter> index_writer;
};

} // namespace lintdb
