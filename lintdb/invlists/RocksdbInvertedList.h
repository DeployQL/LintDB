#ifndef LINTDB_ROCKSDBINVERTEDLIST_H
#define LINTDB_ROCKSDBINVERTEDLIST_H

#include <glog/logging.h>
#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include <rocksdb/utilities/optimistic_transaction_db.h>
#include <iostream>
#include <memory>
#include <string>
#include "lintdb/constants.h"
#include "lintdb/exception.h"
#include "lintdb/invlists/ContextIterator.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/version.h"

namespace lintdb {

/**
 * RocksdbInvertedList stores a slim version of the inverted list. There is no
 * data associated with each token, only the document id as part of the key.
 *
 * This inverted list is only capable of telling us what documents are
 * associated with what centroids.
 */
struct RocksdbInvertedList : public InvertedList {
    RocksdbInvertedList(
            std::shared_ptr<rocksdb::DB> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
            const Version& version);

    void remove(
            const uint64_t tenant,
            std::vector<idx_t> ids,
            const uint8_t field,
            const DataType data_type,
            const std::vector<FieldType> field_types) override;
    void merge(rocksdb::DB* db, std::vector<rocksdb::ColumnFamilyHandle*>& cfs)
            override;

    std::vector<idx_t> get_mapping(const uint64_t tenant, idx_t id)
            const override;

    [[nodiscard]] std::unique_ptr<Iterator> get_iterator(
            const std::string& prefix) const override;

    std::unique_ptr<ContextIterator> get_context_iterator(
            const uint64_t tenant,
            const uint8_t field_id) const override;

   protected:
    Version version;
    std::shared_ptr<rocksdb::DB> db_;
    std::vector<rocksdb::ColumnFamilyHandle*>& column_families;
};

} // namespace lintdb

#endif // LINTDB_ROCKSDBINVERTEDLIST_H
