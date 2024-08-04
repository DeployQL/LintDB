#ifndef LINTDB_ROCKSDB_LIST_H
#define LINTDB_ROCKSDB_LIST_H

#include <glog/logging.h>
#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include <rocksdb/utilities/optimistic_transaction_db.h>
#include <iostream>
#include <memory>
#include <string>
#include "lintdb/constants.h"
#include "lintdb/invlists/ForwardIndexIterator.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/version.h"

namespace lintdb {

struct RocksdbForwardIndex : public ForwardIndex {
    RocksdbForwardIndex(
            std::shared_ptr<rocksdb::DB> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
            const Version& version);

    void remove(const uint64_t tenant, std::vector<idx_t> ids) override;

    void merge(rocksdb::DB* db, std::vector<rocksdb::ColumnFamilyHandle*>& cfs)
            override;

    std::vector<std::map<uint8_t, SupportedTypes>> get_metadata(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const override;

    std::unique_ptr<ForwardIndexIterator> get_iterator(
            const uint64_t tenant,
            idx_t column_index) const override;

   protected:
    Version version;
    std::shared_ptr<rocksdb::DB> db_;
    std::vector<rocksdb::ColumnFamilyHandle*>& column_families;
};

} // namespace lintdb

#endif