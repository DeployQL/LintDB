#pragma once

#include <string>
#include <rocksdb/slice.h>
#include <rocksdb/db.h>
#include "lintdb/invlists/keys.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/constants.h"

namespace lintdb {
class ContextIterator {
public:
    ContextIterator(
        const std::shared_ptr<rocksdb::DB> db,
        rocksdb::ColumnFamilyHandle* column_family,
        const uint64_t tenant,
        const uint8_t field): tenant(tenant), field(field) {

        if(!column_family) {
            throw std::runtime_error("Column family not found");
        }
        cf = column_family->GetID();
        prefix = lintdb::ContextKey{tenant, field, 0, true}.serialize();

        prefix_slice = rocksdb::Slice(this->prefix);
        auto options = rocksdb::ReadOptions();

        this->it = std::unique_ptr<rocksdb::Iterator>(
                db->NewIterator(options, column_family));
        it->Seek(this->prefix);
    }

    bool has_next() {
        if (!has_read_key) {
            bool is_valid = it->Valid();
            if (!is_valid) {
                return false;
            }

            this->current_key = lintdb::ContextKey::from_slice(it->key());
            if (current_key.tenant != tenant || current_key.field != field) {
                return false;
            }
        }

        has_read_key = true;
        return true;
    }

    void advance(const idx_t doc_id) {
        std::string expected_key = ContextKey{tenant, field, doc_id}.serialize();
        it->Seek(rocksdb::Slice(expected_key));
        has_read_key = false;
    }

    void next() {
        it->Next();
        has_read_key = false;
    }

    ContextKey get_key() const {
        return current_key;
    }

    std::string get_value() const {
        return it->value().ToString();
    }

    unique_ptr<rocksdb::Iterator> it;

    protected:
    lintdb::column_index_t cf;
    string prefix;
    string end_key;
    rocksdb::Slice prefix_slice;
    ContextKey current_key;

    bool has_read_key;
    const uint64_t tenant;
    const uint8_t field;
};

} // lintdb
