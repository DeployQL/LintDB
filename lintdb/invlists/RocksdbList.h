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
#include "lintdb/invlists/util.h"
#include "lintdb/schema/util.h"
#include "lintdb/invlists/Iterator.h"

namespace lintdb {

struct RocksDBIterator : public Iterator {
    RocksDBIterator(
            std::shared_ptr<rocksdb::DB> db,
            rocksdb::ColumnFamilyHandle* column_family,
            const uint64_t tenant,
            const idx_t inverted_list);

    bool has_next() override {
        bool is_valid = it->Valid();
        if(!is_valid) {
            return false;
        }
        this->current_key = Key::from_slice(it->key());
        if (current_key.tenant != tenant || current_key.inverted_list_id != inverted_index) {
            return false;
        }

        return true;
    }

    void next() override {
        it->Next();
    }

    Key get_key() const override {
        return current_key;
    }

    std::unique_ptr<rocksdb::Iterator> it;

   private:
    column_index_t cf;
    std::string prefix;
    std::string end_key;
    rocksdb::Slice prefix_slice;
    Key current_key;

    const idx_t tenant;
    const idx_t inverted_index;

};

template<typename DBType>
struct RocksDBInvertedList : public InvertedList, public ForwardIndex {
    RocksDBInvertedList(
            std::shared_ptr<DBType> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families);

    void add(const uint64_t tenant, std::unique_ptr<EncodedDocument> docs) override;
    void remove(const uint64_t tenant, std::vector<idx_t> ids) override;

    void merge(std::shared_ptr<rocksdb::DB> db, std::vector<rocksdb::ColumnFamilyHandle*> cfs) override;

    void delete_entry(idx_t list_no,const uint64_t tenant, idx_t id) override;

    std::unique_ptr<Iterator> get_iterator(
        const uint64_t tenant,
        const idx_t inverted_list) const override;
            
    std::vector<std::unique_ptr<DocumentCodes>> get_codes(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const override;
    std::vector<std::unique_ptr<DocumentResiduals>> get_residuals(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const override;
    std::vector<idx_t> get_mapping(const uint64_t tenant, idx_t id) const override;
    std::vector<std::unique_ptr<DocumentMetadata>> get_metadata(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const override;

    protected:
    std::shared_ptr<DBType> db_;
    std::vector<rocksdb::ColumnFamilyHandle*>& column_families;
};

struct WritableRocksDBInvertedList : public RocksDBInvertedList<rocksdb::OptimisticTransactionDB> {
    WritableRocksDBInvertedList(
            std::shared_ptr<rocksdb::OptimisticTransactionDB> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families);

    /**
     * Add transactionally adds data to the database.
     * 
    */
    void add(const uint64_t tenant, std::unique_ptr<EncodedDocument> docs) override;

};

struct ReadOnlyRocksDBInvertedList : public RocksDBInvertedList<rocksdb::DB> {
    ReadOnlyRocksDBInvertedList(
            std::shared_ptr<rocksdb::DB> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families);

    void add(const uint64_t tenant, std::unique_ptr<EncodedDocument> docs) override;
    void remove(const uint64_t tenant, std::vector<idx_t> ids) override;

    void merge(std::shared_ptr<rocksdb::DB> db, std::vector<rocksdb::ColumnFamilyHandle*> cfs) override;

    void delete_entry(idx_t list_no,const uint64_t tenant, idx_t id) override;
};

} // namespace lintdb

#endif