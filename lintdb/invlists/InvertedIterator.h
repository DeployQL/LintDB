#pragma once

#include <glog/logging.h>
#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include <rocksdb/utilities/optimistic_transaction_db.h>
#include <iostream>
#include <memory>
#include <string>
#include "lintdb/constants.h"
#include "lintdb/utils/exception.h"
#include "lintdb/invlists/ContextIterator.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/invlists/KeyBuilder.h"
#include "lintdb/version.h"

namespace lintdb {

struct RocksDBIterator : public lintdb::Iterator {
    RocksDBIterator(
            std::shared_ptr<rocksdb::DB> db,
            rocksdb::ColumnFamilyHandle* column_family,
            const std::string& prefix);

    bool is_valid() override {
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

            current_key = InvertedIndexKey(key_str);
        }

        has_read_key = true;
        return true;
    }

    void next() override {
        it->Next();
        has_read_key = false;
    }

    InvertedIndexKey get_key() const override {
        return current_key;
    }

    string get_value() const override {
        return it->value().ToString();
    }

    std::unique_ptr<rocksdb::Iterator> it;

   protected:
    lintdb::column_index_t cf;
    string prefix;
    string end_key;
    rocksdb::Slice prefix_slice;
    InvertedIndexKey current_key;

    bool has_read_key;
};

} // namespace lintdb