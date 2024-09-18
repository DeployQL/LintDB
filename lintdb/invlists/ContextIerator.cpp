#include "ContextIterator.h"
#include "KeyBuilder.h"
#include <rocksdb/db.h>
#include <rocksdb/slice.h>

namespace lintdb {

ContextIterator::ContextIterator(
        const std::shared_ptr<rocksdb::DB> db,
        rocksdb::ColumnFamilyHandle* column_family,
        const uint64_t tenant,
        const uint8_t field)
        : tenant(tenant), field(field), has_read_key(false) {
    if (!column_family) {
        throw std::runtime_error("Column family not found");
    }
    cf = column_family->GetID();
    KeyBuilder kb;
    prefix = kb.add(tenant).add(field).build();

    auto options = rocksdb::ReadOptions();

    this->it = std::unique_ptr<rocksdb::Iterator>(
            db->NewIterator(options, column_family));
    it->Seek(this->prefix);
}

bool ContextIterator::is_valid() {
    if (!has_read_key) {
        bool is_valid = it->Valid();
        if (!is_valid) {
            return false;
        }

        auto key = it->key();
        std::string key_str = key.ToString();
        if (key_str.compare(0, prefix.size(), prefix) != 0) {
            return false;
        }
        this->current_key = ContextKey(key_str);
    }

    has_read_key = true;
    return true;
}

void ContextIterator::advance(const idx_t doc_id) {
    KeyBuilder kb;

    std::string expected_key =
            kb.add(tenant).add(field).add(doc_id).build();
    it->Seek(rocksdb::Slice(expected_key));
    has_read_key = false;
}

void ContextIterator::next() {
    it->Next();
    has_read_key = false;
}

ContextKey ContextIterator::get_key() const {
    return current_key;
}

std::string ContextIterator::get_value() const {
    return it->value().ToString();
}

} // namespace lintdb