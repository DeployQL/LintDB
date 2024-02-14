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
        const size_t list_no
    ) : Iterator(), it(db.NewIterator(rocksdb::ReadOptions(), column_family)), list_no(list_no), cf(column_family->GetID()) {
        Key start_key = Key{kDefaultTenant, list_no, 0, true};
        it->Seek(rocksdb::Slice(start_key.serialize()));
    }

    bool has_next() const override {
        return it->Valid();

    }
    void next() override {
        it->Next();
    }

    EncodedDocument get() const override {
        // value is going to be a byte array. We should be able to convert it to EncodedDocument.
        rocksdb::Slice val = it->value();
        // key is also a byte array. We need to read it in and convert it to the id.
        rocksdb::Slice key = it->key();
        Key k = Key::from_slice(key);

        if (cf == kForwardColumnIndex) {
            auto doc = GetForwardIndexDocument(val.data());
            const code_t* codes = nullptr;
            return EncodedDocument(
                codes,
                0,
                doc->residuals()->data(),
                doc->residuals()->size(),
                doc->num_tokens(),
                k.id,
                doc->doc_id()->str());
        } else {
            auto doc = GetInvertedIndexDocument(val.data());
            const residual_t* residuals = nullptr;
            return EncodedDocument(
                doc->codes()->data(),
                doc->codes()->size(),
                residuals,
                0,
                doc->codes()->size(), // TODO(mbarta): this won't always hold. we need to be better about abstracting this.
                k.id,
                ""
            );
        }

    }

    std::unique_ptr<rocksdb::Iterator> it;
    private:
    size_t list_no;
    column_index_t cf;
};

struct RocksDBInvertedList: public InvertedList {
    RocksDBInvertedList(rocksdb::DB& db, std::vector<rocksdb::ColumnFamilyHandle*>& column_families);

    void add(
            size_t list_no,
            std::unique_ptr<EncodedDocument>& docs) override;

    void delete_entry(
            size_t list_no,
            idx_t id) override;

    std::unique_ptr<Iterator> get_iterator(size_t list_no) const override;
    std::vector<std::unique_ptr<EncodedDocument>> get(std::vector<idx_t> ids) const override;

    private:
    rocksdb::DB& db_;
    std::vector<rocksdb::ColumnFamilyHandle*>& column_families;
};

}

#endif