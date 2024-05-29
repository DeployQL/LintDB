#pragma once

#include <memory>
#include <vector>
#include <rocksdb/db.h>
#include <cstdint>
#include "lintdb/api.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/Iterator.h"
#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include "lintdb/version.h"
#include "lintdb/invlists/RocksdbInvertedList.h"


namespace lintdb {
struct RocksDBIteratorV2 : public RocksDBIterator {
    RocksDBIteratorV2(
            shared_ptr<rocksdb::DB> db,
            rocksdb::ColumnFamilyHandle* column_family,
            uint64_t tenant,
            idx_t inverted_list);

    Key get_key() const override {
        throw LintDBException("not supported in RocksdbInvertedList. Use RocksdbInvertedListV2");
    }

    bool has_next() override {
        bool is_valid = it->Valid();
        if(!is_valid) {
            return false;
        }
        this->current_key = TokenKey::from_slice(it->key());
        if (current_key.tenant != tenant || current_key.inverted_list_id != inverted_index) {
            return false;
        }

        return true;
    }

    TokenKey get_token_key() const override {
        return current_key;
    }

   private:
    TokenKey current_key;
};
    /**
     * RocksdbInvertedListV2 stores more data into the inverted list than its
     * predecessor. We realized that we also want the token codes when we're
     * searching, so we may as well store them into the inverted index.
     *
     * This list expects TokenKeys, which stores keys that point to the doc token's codes.
    */
    struct RocksdbInvertedListV2: public RocksdbInvertedList {
        RocksdbInvertedListV2(
            std::shared_ptr<rocksdb::DB> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
            Version& version);

        void add(uint64_t tenant, std::unique_ptr<EncodedDocument> doc) override;

        [[nodiscard]]
        std::unique_ptr<Iterator> get_iterator(
                uint64_t tenant,
                idx_t inverted_list) const override;
    };
}