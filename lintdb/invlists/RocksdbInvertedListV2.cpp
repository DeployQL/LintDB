#include "lintdb/invlists/RocksdbInvertedListV2.h"
#include <glog/logging.h>
#include "lintdb/constants.h"

namespace lintdb {

RocksDBIteratorV2::RocksDBIteratorV2(
        shared_ptr<rocksdb::DB> db,
        rocksdb::ColumnFamilyHandle* column_family,
        const uint64_t tenant,
        const idx_t inverted_list)
        : RocksDBIterator(db, column_family, tenant, inverted_list) {}

RocksdbInvertedListV2::RocksdbInvertedListV2(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
        Version& version
    ) : RocksdbInvertedList(db, column_families, version) {}

    void RocksdbInvertedListV2::add(const uint64_t tenant, std::unique_ptr<EncodedDocument> doc) {
        rocksdb::WriteOptions wo;
        std::unordered_set<idx_t> unique_coarse_idx(
                doc->codes.begin(), doc->codes.end());
        VLOG(100) << "Unique coarse indexes: " << unique_coarse_idx.size();

        auto inverted_data = doc->serialize_inverted_data();

        // store ivf -> doc mapping.
        // serialized data is defined in EncodedDocument.cpp
        for (const auto& data: inverted_data) {
            auto key = TokenKey{tenant, data.key, doc->id, data.token_id};
            std::string k_string = key.serialize();

            rocksdb::Status status = db_->Put(
                    wo,
                    column_families[kIndexColumnIndex],
                    rocksdb::Slice(k_string),
                    rocksdb::Slice(data.value)
            );
            assert(status.ok());
            VLOG(100) << "Added document with id: " << doc->id
                      << " to inverted list " << data.key;
        }
    }

    std::unique_ptr<Iterator> RocksdbInvertedListV2::get_iterator(
            const uint64_t tenant,
            const idx_t inverted_list) const {
        return std::make_unique<RocksDBIteratorV2>(RocksDBIteratorV2(
                db_, column_families[kIndexColumnIndex], tenant, inverted_list));
    }


}