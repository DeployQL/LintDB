#include "DocValue.h"
#include <vector>
#include <queue>
#include "lintdb/schema/DataTypes.h"
#include "decode.h"

namespace lintdb {
DocValue decode_vectors(const lintdb::QueryContext& context, const lintdb::DocValue& doc_value) {
    if (doc_value.unread_value) {
        return doc_value;
    }
    switch (doc_value.type) {
        case lintdb::QUANTIZED_TENSOR: {
            std::string field = context.getFieldMapper()->getFieldName(doc_value.field_id);
            // check if field has a quantizer.
            if (!context.getQuantizer(field)) {
                return doc_value;
            }

            auto quantizer = context.getQuantizer(field);

            std::vector<residual_t> quantized = std::get<lintdb::QuantizedTensor>(doc_value.get_value());
            size_t dim = context.getFieldMapper()->getFieldDimensions(doc_value.field_id);
            size_t num_vectors = quantized.size() / quantizer->code_size();

            std::vector<float> tensor(num_vectors * dim, 0);
            quantizer->sa_decode(num_vectors, quantized.data(), tensor.data());

            return {tensor, doc_value.field_id, lintdb::TENSOR};
        }
        default:
            return doc_value;
    }
}
}