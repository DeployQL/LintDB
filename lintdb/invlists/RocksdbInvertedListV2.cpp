#include "lintdb/invlists/RocksdbInvertedListV2.h"
#include <glog/logging.h>
#include "lintdb/assert.h"
#include "lintdb/constants.h"

namespace lintdb {

RocksdbInvertedListV2::RocksdbInvertedListV2(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
        Version& version)
        : RocksdbInvertedList(db, column_families, version) {}

void RocksdbInvertedListV2::add(const uint64_t tenant, EncodedDocument* doc) {
    auto inverted_data = doc->serialize_inverted_data();
    rocksdb::WriteOptions wo;
    // store ivf -> doc mapping.
    // serialized data is defined in EncodedDocument.cpp
    for (const auto& data : inverted_data) {
        auto token_id = data.token_id;
        auto key = TokenKey{tenant, data.key, doc->id, token_id, false};
        std::string k_string = key.serialize();

        const char* kk_string = reinterpret_cast<const char*>(k_string.data());
        std::string v_string = data.value;
        const char* vv_string = reinterpret_cast<const char*>(v_string.data());
        assert(!v_string.empty());

        rocksdb::Status status = db_->Put(
                wo,
                column_families[kIndexColumnIndex],
                rocksdb::Slice(kk_string, k_string.size()),
                rocksdb::Slice(vv_string, v_string.size()));
        assert(status.ok());
        VLOG(10) << "Added document with id: " << doc->id << " "
                 << data.token_id << " to inverted list " << data.key;
    }

    std::unordered_set<idx_t> unique_coarse_idx(
            doc->codes.begin(), doc->codes.end());

    ForwardIndexKey forward_key = ForwardIndexKey{tenant, doc->id};
    auto fks = forward_key.serialize();
    std::vector<idx_t> unique_coarse_idx_vec(
            unique_coarse_idx.begin(), unique_coarse_idx.end());
    auto mapping_ptr = create_doc_mapping(
            unique_coarse_idx_vec.data(), unique_coarse_idx_vec.size());
    auto mapping_status = db_->Put(
            wo,
            column_families[kMappingColumnIndex],
            rocksdb::Slice(fks),
            rocksdb::Slice(
                    reinterpret_cast<const char*>(
                            mapping_ptr->GetBufferPointer()),
                    mapping_ptr->GetSize()));
    LINTDB_THROW_IF_NOT(mapping_status.ok());
}
} // namespace lintdb