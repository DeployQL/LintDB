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
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/invlists/keys.h"
#include "lintdb/schema/util.h"
#include "lintdb/version.h"
#include "lintdb/invlists/ForwardIndexIterator.h"

namespace lintdb {

struct RocksdbForwardIndex : public ForwardIndex {
    RocksdbForwardIndex(
            std::shared_ptr<rocksdb::DB> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
            const Version& version);

    // store_codes is used to determine whether to store the codes in the
    // forward index. ColBERT uses the forward index for codes, but XTR does
    // not.
    void add(
            const uint64_t tenant,
            EncodedDocument* doc,
            bool store_codes = true);
    void remove(const uint64_t tenant, std::vector<idx_t> ids) override;

    void merge(rocksdb::DB* db, std::vector<rocksdb::ColumnFamilyHandle*>& cfs)
            override;

    std::vector<std::unique_ptr<DocumentCodes>> get_codes(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const override;
    std::vector<std::unique_ptr<DocumentResiduals>> get_residuals(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const override;
    std::vector<std::unique_ptr<DocumentMetadata>> get_metadata(
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