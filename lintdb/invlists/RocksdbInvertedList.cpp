#include "RocksdbInvertedList.h"
#include <glog/logging.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>
#include <iostream>
#include "lintdb/assert.h"
#include "lintdb/constants.h"
#include "lintdb/exception.h"
#include "lintdb/invlists/RocksdbForwardIndex.h"
#include "lintdb/schema/forward_index_generated.h"
#include "lintdb/schema/util.h"
#include "lintdb/invlists/ContextIterator.h"
#include "InvertedIterator.h"

namespace lintdb {

    RocksdbInvertedList::RocksdbInvertedList(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
        const Version& version)
        : version(version), db_(db), column_families(column_families) {}

void RocksdbInvertedList::remove(
        const uint64_t tenant,
        std::vector<idx_t> ids) {
    for (idx_t id : ids) {
        auto id_map = this->get_mapping(tenant, id);
        // delete from the inverse index.
        rocksdb::ReadOptions ro;
        for (auto idx : id_map) {
            Key key = Key{tenant, idx, id};
            std::string k_string = key.serialize();
            rocksdb::WriteOptions wo;
            rocksdb::Status status = db_->Delete(
                    wo,
                    column_families[kIndexColumnIndex],
                    rocksdb::Slice(k_string));
            assert(status.ok());
        }
    }
}

std::vector<idx_t> RocksdbInvertedList::get_mapping(
        const uint64_t tenant,
        idx_t id) const {
    auto key = ForwardIndexKey{tenant, id};
    auto serialized_key = key.serialize();
    rocksdb::ReadOptions ro;
    std::string value;
    rocksdb::Status status = db_->Get(
            ro,
            column_families[kMappingColumnIndex],
            rocksdb::Slice(serialized_key),
            &value);

    if (status.ok()) {
        auto mapping = GetDocumentClusterMapping(value.data());
        std::vector<idx_t> idxs;
        for (size_t i = 0; i < mapping->centroids()->size(); i++) {
            idxs.push_back(mapping->centroids()->Get(i));
        }
        return idxs;
    } else {
        LOG(WARNING) << "Could not find mapping for doc id: " << id;
        return {};
    }
}

// merge uses the index and mapping column families.
void RocksdbInvertedList::merge(
        rocksdb::DB* db,
        std::vector<rocksdb::ColumnFamilyHandle*>& cfs) {
    // very weak check to make sure the column families are the same.
    LINTDB_THROW_IF_NOT(cfs.size() == column_families.size());

    auto cf = cfs[kIndexColumnIndex];
    rocksdb::ReadOptions ro;
    auto it = db->NewIterator(ro, cf);
    it->SeekToFirst();
    rocksdb::WriteOptions wo;
    while (it->Valid()) {
        auto key = it->key();
        auto value = it->value();
        auto status =
                db_->Put(wo, column_families[kIndexColumnIndex], key, value);
        assert(status.ok());
        it->Next();
    }

    auto map_cf = cfs[kMappingColumnIndex];
    auto map_it = db->NewIterator(ro, map_cf);
    map_it->SeekToFirst();
    while (map_it->Valid()) {
        auto key = map_it->key();
        auto value = map_it->value();
        auto status =
                db_->Put(wo, column_families[kMappingColumnIndex], key, value);
        assert(status.ok());
        map_it->Next();
    }
}

std::unique_ptr<Iterator> RocksdbInvertedList::get_iterator(
        const uint64_t tenant,
        const uint8_t field,
        const idx_t inverted_list) const {
    return std::make_unique<RocksDBIterator>(
            db_, column_families[kIndexColumnIndex], tenant, field, inverted_list);
}

std::unique_ptr<ContextIterator> RocksdbInvertedList::get_context_iterator(
        const uint64_t tenant,
        const uint8_t field_id
) const {
    return std::make_unique<ContextIterator>(db_, column_families[kCodesColumnIndex], tenant, field_id);
}
} // namespace lintdb