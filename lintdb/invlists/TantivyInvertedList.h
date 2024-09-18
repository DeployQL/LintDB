#pragma once

#include "third_party/tantivy/tantivy-binding/include/tantivy-binding.h"
#include "lintdb/invlists/InvertedList.h"

namespace lintdb {

/**
 * TantivyInvertedList is an InvertedList implementation that uses Tantivy
 * for sparse inverted indices.
 */
class TantivyInvertedList: public InvertedList {
   public:
    void remove(
            const uint64_t tenant,
            std::vector<idx_t> ids,
            const uint8_t field,
            const DataType data_type,
            const std::vector<FieldType> field_types) override;
    void merge(rocksdb::DB* db, std::vector<rocksdb::ColumnFamilyHandle*>& cfs)
            override;
    std::unique_ptr<Iterator> get_iterator(const string& prefix) const override;
    std::unique_ptr<ContextIterator> get_context_iterator(
            const uint64_t tenant,
            const uint8_t field_id) const override;
    std::vector<idx_t> get_mapping(const uint64_t tenant, idx_t id)
            const override;
};

} // namespace lintdb
