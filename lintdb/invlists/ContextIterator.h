#pragma once

#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/iterator.h>
#include <string>
#include "lintdb/constants.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/invlists/KeyBuilder.h"

namespace lintdb {

class ContextIterator {
   public:
    ContextIterator(
            const std::shared_ptr<rocksdb::DB> db,
            rocksdb::ColumnFamilyHandle* column_family,
            const uint64_t tenant,
            const uint8_t field);

    bool is_valid();
    void advance(const idx_t doc_id);
    void next();
    ContextKey get_key() const;
    std::string get_value() const;

   private:
    std::unique_ptr<rocksdb::Iterator> it;
    lintdb::column_index_t cf;
    std::string prefix;
    std::string end_key;
    rocksdb::Slice prefix_slice;
    ContextKey current_key;
    bool has_read_key;
    const uint64_t tenant;
    const uint8_t field;
};


} // namespace lintdb
