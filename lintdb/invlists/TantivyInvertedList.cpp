#include "TantivyInvertedList.h"

namespace lintdb {

std::unique_ptr<Iterator> TantivyInvertedList::get_iterator(
        const string& prefix) const {
    return std::unique_ptr<Iterator>();
}

std::unique_ptr<ContextIterator> TantivyInvertedList::get_context_iterator(
        const uint64_t tenant,
        const uint8_t field_id) const {
    return std::unique_ptr<ContextIterator>();
}

std::vector<idx_t> TantivyInvertedList::get_mapping(
        const uint64_t tenant,
        idx_t id) const {
    return std::vector<idx_t>();
}

void TantivyInvertedList::merge(
        rocksdb::DB* db,
        std::vector<rocksdb::ColumnFamilyHandle*>& cfs) {}

void TantivyInvertedList::remove(
        const uint64_t tenant,
        std::vector<idx_t> ids,
        const uint8_t field,
        const DataType data_type,
        const std::vector<FieldType> field_types) {}
} // namespace lintdb