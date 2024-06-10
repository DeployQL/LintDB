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
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/invlists/keys.h"
#include "lintdb/schema/util.h"
#include "lintdb/version.h"
#include "lintdb/exception.h"

namespace lintdb {

struct RocksDBIterator : public lintdb::Iterator {
    RocksDBIterator(
            shared_ptr<rocksdb::DB> db,
            rocksdb::ColumnFamilyHandle* column_family,
            const uint64_t tenant,
            const idx_t inverted_list);

    bool has_next() override {
        bool is_valid = it->Valid();
        if(!is_valid) {
            return false;
        }
        this->current_key = lintdb::TokenKey::from_slice(it->key());

        if (current_key.tenant != tenant || current_key.inverted_list_id != inverted_index) {
            return false;
        }

        return true;
    }

    void next() override {
        it->Next();
    }

    TokenKey get_key() const override {
        return current_key;
    }

    PartialDocumentCodes get_value() const override {
        auto value = it->value().ToString();
        auto id = get_key().doc_id;
        return PartialDocumentCodes::deserialize(id, value);
    }

    unique_ptr<rocksdb::Iterator> it;

   protected:
    lintdb::column_index_t cf;
    string prefix;
    string end_key;
    rocksdb::Slice prefix_slice;
    TokenKey current_key;

    const idx_t tenant;
    const idx_t inverted_index;

};

/**
 * RocksdbInvertedList stores a slim version of the inverted list. There is no data
 * associated with each token, only the document id as part of the key.
 *
 * This inverted list is only capable of telling us what documents are associated
 * with what centroids.
 */
struct RocksdbInvertedList: public InvertedList {
    RocksdbInvertedList(
            std::shared_ptr<rocksdb::DB> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
            Version& version);

    void add(uint64_t tenant, EncodedDocument* doc)
            override;
    void remove(uint64_t tenant, std::vector<idx_t> ids) override;
    void merge(
            rocksdb::DB* db,
            std::vector<rocksdb::ColumnFamilyHandle*> cfs) override;

    std::vector<idx_t> get_mapping(const uint64_t tenant, idx_t id) const override;

    [[nodiscard]]
    unique_ptr<Iterator> get_iterator(
            uint64_t tenant,
            idx_t inverted_list) const override;

   protected:
    Version version;
    std::shared_ptr<rocksdb::DB> db_;
    std::vector<rocksdb::ColumnFamilyHandle*>& column_families;
};

} // namespace lintdb

#endif // LINTDB_ROCKSDBINVERTEDLIST_H
