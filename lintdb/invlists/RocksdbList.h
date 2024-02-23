#ifndef LINTDB_ROCKSDB_LIST_H
#define LINTDB_ROCKSDB_LIST_H

#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include "lintdb/invlists/util.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/schema/util.h"
#include "lintdb/constants.h"
#include <memory>
#include <glog/logging.h>
#include <iostream>

namespace lintdb {

struct RocksDBIterator : public Iterator {
    RocksDBIterator(
        rocksdb::DB& db,
        rocksdb::ColumnFamilyHandle* column_family,
        const std::string& start_key,
        const std::string& end_key
    ) : Iterator(), cf(column_family->GetID()), end_slice(end_key) {

        rocksdb::Slice prefix(start_key);

        auto options = rocksdb::ReadOptions();
        options.iterate_upper_bound = &this->end_slice;

        this->it = std::unique_ptr<rocksdb::Iterator>(db.NewIterator(options, column_family));
        it->Seek(prefix);
    }

    bool has_next() const override {
        return it->Valid();
    }

    void next() override {
        it->Next();
    }


    Key get_key() const override {
        rocksdb::Slice key = it->key();
        return Key::from_slice(key);
    }

    std::unique_ptr<rocksdb::Iterator> it;
    private:
    column_index_t cf;
    rocksdb::Slice end_slice;
};

struct RocksDBInvertedList: public InvertedList {
    RocksDBInvertedList(rocksdb::DB& db, std::vector<rocksdb::ColumnFamilyHandle*>& column_families);

    void add(std::unique_ptr<EncodedDocument> docs) override;
    void remove(std::vector<idx_t> ids) override;

    void delete_entry(
            idx_t list_no,
            idx_t id) override;

    std::unique_ptr<Iterator> get_iterator(const std::string& start, const std::string& end) const override;
    std::vector<std::unique_ptr<EncodedDocument>> get_codes(std::vector<idx_t> ids) const override;
    std::vector<std::unique_ptr<EncodedDocument>> get_residuals(std::vector<idx_t> ids) const override;
    std::vector<idx_t> get_mapping(idx_t id) const override;

    private:
    rocksdb::DB& db_;
    std::vector<rocksdb::ColumnFamilyHandle*>& column_families;
};

}

#endif