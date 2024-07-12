#include "InvertedIterator.h"
#include "lintdb/invlists/ContextIterator.h"
#include "lintdb/schema/util.h"
#include "lintdb/schema/forward_index_generated.h"
#include "lintdb/invlists/RocksdbForwardIndex.h"
#include "lintdb/exception.h"
#include "lintdb/constants.h"
#include "lintdb/assert.h"
#include <unordered_set>
#include <iostream>
#include <rocksdb/utilities/transaction.h>
#include <rocksdb/slice.h>
#include <glog/logging.h>
#include "RocksdbInvertedList.h"

lintdb::RocksDBIterator::RocksDBIterator(
        shared_ptr<rocksdb::DB> db,
        rocksdb::ColumnFamilyHandle* column_family,
        const uint64_t tenant,
        const uint8_t field,
        const idx_t inverted_list)
        : Iterator(), has_read_key(false), tenant(tenant), field(field), inverted_index(inverted_list) {
    cf = column_family->GetID();
    prefix = TokenKey{tenant, field, inverted_list, 0, 0, true}.serialize();

    prefix_slice = rocksdb::Slice(this->prefix);
    auto options = rocksdb::ReadOptions();

    this->it = unique_ptr<rocksdb::Iterator>(
            db->NewIterator(options, column_family));
    it->Seek(this->prefix);
}