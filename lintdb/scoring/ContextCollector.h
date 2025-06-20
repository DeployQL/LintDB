#pragma once

#include <vector>
#include <memory>
#include <string>
#include "lintdb/query/QueryContext.h"
#include "lintdb/query/DocValue.h"
#include "lintdb/invlists/ContextIterator.h"
#include "lintdb/schema/DocEncoder.h"

namespace lintdb {

    class ContextCollector {
       public:
        ContextCollector() = default;

        void add_field(const QueryContext& context, const std::string& field);
        std::vector<DocValue> get_context_values(const idx_t doc_id) const;

       private:
        std::vector<std::string> context_fields;
        std::vector<uint8_t> context_field_ids;
        std::vector<DataType> context_data_types;
        std::vector<std::unique_ptr<ContextIterator>> context_iterators;
    };

} // namespace lintdb