#include "ForwardIndexIterator.h"
#include <memory>

namespace lintdb {
ForwardIndexIterator::ForwardIndexIterator(
        std::shared_ptr<rocksdb::DB> db,
        rocksdb::ColumnFamilyHandle* column_family,
        const uint64_t tenant)
        : tenant(tenant) {
    cf = column_family->GetID();
    KeyBuilder kb;

    prefix = kb.add(tenant).build();

    prefix_slice = rocksdb::Slice(this->prefix);
    auto options = rocksdb::ReadOptions();

    this->it = std::unique_ptr<rocksdb::Iterator>(
            db->NewIterator(options, column_family));
    it->Seek(this->prefix);
}

bool ForwardIndexIterator::has_next() {
    bool is_valid = it->Valid();
    if (!is_valid) {
        return false;
    }
    auto key = it->key().ToString();
    this->current_key = ForwardIndexKey(key);

    if (current_key.tenant() != tenant) {
        return false;
    }

    return true;
}

void ForwardIndexIterator::next() {
    it->Next();
}

ForwardIndexKey ForwardIndexIterator::get_key() const {
    return current_key;
}

std::string ForwardIndexIterator::get_value() const {
    return it->value().ToString();
}
} // namespace lintdb