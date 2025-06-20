#include "ContextCollector.h"
#include <glog/logging.h>
#include <algorithm>
#include <rocksdb/iterator.h>

namespace lintdb {

void ContextCollector::add_field(const QueryContext& context, const std::string& field) {
    context_fields.push_back(field);

    uint8_t colbert_field_id =
            context.getFieldMapper()->getFieldID(context.colbert_context);
    context_field_ids.push_back(colbert_field_id);

    bool is_colbert = false;
    auto field_types = context.getFieldMapper()->getFieldTypes(colbert_field_id);
    /**
     * This is a pretty big hack because we modify the ColBERT fields internally. A user passes in
     * a tensor data type, and we process it distinctly for colbert and reset it to be datatype::colbert.
     *
     * A solution is to stop modifying datatypes internally, or we could expose ColBERT
     * as a datatype. However, our colbert storage is meant to be internal.
     */
    if (std::find(field_types.begin(), field_types.end(), FieldType::Colbert) != field_types.end()) {
        is_colbert = true;
    }
    if (!is_colbert) {
        context_data_types.push_back(context.getFieldMapper()->getDataType(colbert_field_id));
    } else {
        context_data_types.push_back(DataType::COLBERT);
    }

    std::unique_ptr<ContextIterator> it = context.getIndex()->get_context_iterator(
            context.getTenant(), colbert_field_id);

    context_iterators.push_back(std::move(it));
}

std::vector<DocValue> ContextCollector::get_context_values(const idx_t doc_id) const {
    std::vector<DocValue> results;
    results.reserve(context_iterators.size());

    for(int i=0; i < context_iterators.size(); i++) {
        auto it = context_iterators[i].get();
        it->advance(doc_id);

        if(it->is_valid() && it->get_key().doc_id() == doc_id) {
            std::string context_str = it->get_value();
            SupportedTypes colbert_context =
                    DocEncoder::decode_supported_types(context_str);

            // create DocValues for the context info.
            uint8_t colbert_field_id = context_field_ids[i];
            results.emplace_back(colbert_context, colbert_field_id, context_data_types[i]);
        } else {
            LOG(WARNING) << "No context found for doc_id: " << doc_id << " field: " << context_fields[i];
        }
    }

    return results;
}

} // namespace lintdb