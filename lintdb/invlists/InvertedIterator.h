#pragma once

#include <glog/logging.h>
#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include <rocksdb/utilities/optimistic_transaction_db.h>
#include <iostream>
#include <memory>
#include <string>
#include "lintdb/constants.h"
#include "lintdb/exception.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/invlists/keys.h"
#include "lintdb/schema/util.h"
#include "lintdb/version.h"
#include "lintdb/invlists/ContextIterator.h"

namespace lintdb {

    struct RocksDBIterator : public lintdb::Iterator {
        RocksDBIterator(
                shared_ptr<rocksdb::DB> db,
                rocksdb::ColumnFamilyHandle *column_family,
                const uint64_t tenant,
                const uint8_t field,
                const idx_t inverted_list);

        bool is_valid() override {
            if (!has_read_key) {
                bool is_valid = it->Valid();
                if (!is_valid) {
                    return false;
                }

                this->current_key = lintdb::TokenKey::from_slice(it->key());
                if (current_key.tenant != tenant ||
                    current_key.inverted_list_id != inverted_index) {
                    return false;
                }
            }

            has_read_key = true;
            return true;
        }

        void next() override {
            it->Next();
            has_read_key = false;
        }

        lintdb::TokenKey get_key() const override {
            return current_key;
        }

        string get_value() const override {
            return it->value().ToString();
        }

        unique_ptr<rocksdb::Iterator> it;

    protected:
        lintdb::column_index_t cf;
        string prefix;
        string end_key;
        rocksdb::Slice prefix_slice;
        lintdb::TokenKey current_key;

        bool has_read_key;
        const idx_t tenant;
        const uint8_t field;
        const idx_t inverted_index;
    };

}