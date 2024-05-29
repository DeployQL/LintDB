#include "RocksdbInvertedList.h"
#include <glog/logging.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>
#include <iostream>
#include <unordered_set>
#include "lintdb/assert.h"
#include "lintdb/constants.h"
#include "lintdb/exception.h"
#include "lintdb/invlists/RocksdbForwardIndex.h"
#include "lintdb/schema/forward_index_generated.h"
#include "lintdb/schema/util.h"

namespace lintdb {

RocksDBIterator::RocksDBIterator(
        shared_ptr<rocksdb::DB> db,
        rocksdb::ColumnFamilyHandle* column_family,
        const uint64_t tenant,
        const idx_t inverted_list)
        : lintdb::Iterator(), tenant(tenant), inverted_index(inverted_list) {
    cf = column_family->GetID();
    prefix = lintdb::Key{tenant, inverted_list, 0, true}.serialize();

    prefix_slice = rocksdb::Slice(this->prefix);
    auto options = rocksdb::ReadOptions();

    this->it = unique_ptr<rocksdb::Iterator>(
            db->NewIterator(options, column_family));
    it->Seek(this->prefix);
}

RocksdbInvertedList::RocksdbInvertedList(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
        Version& version)
        : db_(db), column_families(column_families), version(version) {}

void RocksdbInvertedList::add(
        const uint64_t tenant,
        std::unique_ptr<EncodedDocument> doc) {

    rocksdb::WriteOptions wo;
    std::unordered_set<idx_t> unique_coarse_idx(
            doc->codes.begin(), doc->codes.end());
    VLOG(100) << "Unique coarse indexes: " << unique_coarse_idx.size();

    // store ivf -> doc mapping.
    for (const code_t& idx : unique_coarse_idx) {
        Key key = Key{tenant, idx, doc->id};
        std::string k_string = key.serialize();

        rocksdb::Status status = db_->Put(
                wo,
                column_families[kIndexColumnIndex],
                rocksdb::Slice(k_string),
                rocksdb::Slice() // store nothing. we only need the key
                                 // to tell us what documents exist.
        );
        assert(status.ok());
        VLOG(100) << "Added document with id: " << doc->id
                  << " to inverted list " << idx;
    }
}

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

void RocksdbInvertedList::merge(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*> cfs) {
    // very weak check to make sure the column families are the same.
    LINTDB_THROW_IF_NOT(cfs.size() == column_families.size());

#pragma omp for
    for (size_t i = 1; i < cfs.size(); i++) {
        // ignore the default cf at position 0 since we don't use it.
        auto cf = cfs[i];
        rocksdb::ReadOptions ro;
        auto it = db->NewIterator(ro, cf);
        it->SeekToFirst();
        rocksdb::WriteOptions wo;
        while (it->Valid()) {
            auto key = it->key();
            auto value = it->value();
            auto status = db_->Put(wo, column_families[i], key, value);
            assert(status.ok());
            it->Next();
        }

        delete it;
    }
}


unique_ptr<Iterator> RocksdbInvertedList::get_iterator(
        const uint64_t tenant,
        const idx_t inverted_list) const {
    return std::make_unique<RocksDBIterator>(RocksDBIterator(
            db_, column_families[kIndexColumnIndex], tenant, inverted_list));
}
} // namespace lintdb