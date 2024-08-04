#include "InvertedIterator.h"
#include <glog/logging.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>
#include <memory>
#include "lintdb/constants.h"
#include "lintdb/invlists/ContextIterator.h"

lintdb::RocksDBIterator::RocksDBIterator(
        std::shared_ptr<rocksdb::DB> db,
        rocksdb::ColumnFamilyHandle* column_family,
        const std::string& prefix)
        : Iterator(), prefix(prefix), has_read_key(false) {
    cf = column_family->GetID();

    prefix_slice = rocksdb::Slice(this->prefix);
    auto options = rocksdb::ReadOptions();

    this->it = std::unique_ptr<rocksdb::Iterator>(
            db->NewIterator(options, column_family));
    it->Seek(this->prefix);
}