#pragma once

#include "lintdb/invlists/KeyBuilder.h"
#include "lintdb/constants.h"
#include "rocksdb/db.h"
#include <memory>

namespace lintdb {

/**
 * ForwardIndexIterator is an iterator over the forward index.
 *
 * This is somewhat coupled to key format so that we can control
 * iteration. Note that while RocksDB has start and stop option,
 * it was not working as expected. So we are doing it manually.
 */
struct ForwardIndexIterator {
    ForwardIndexIterator(
            std::shared_ptr<rocksdb::DB> db,
            rocksdb::ColumnFamilyHandle* column_family,
            const uint64_t tenant);

    bool has_next();

    void next();

    ForwardIndexKey get_key() const;

    std::string get_value() const;

    std::unique_ptr<rocksdb::Iterator> it;

   protected:
    lintdb::column_index_t cf;
    string prefix;
    string end_key;
    rocksdb::Slice prefix_slice;
    ForwardIndexKey current_key;

    const idx_t tenant;
};

} // namespace lintdb
