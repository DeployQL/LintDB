#include "lintdb/invlists/RocksdbList.h"
#include <glog/logging.h>
#include <iostream>
#include <unordered_set>
#include "lintdb/assert.h"
#include "lintdb/constants.h"
#include "lintdb/schema/forward_index_generated.h"
#include "lintdb/schema/util.h"

namespace lintdb {
RocksDBInvertedList::RocksDBInvertedList(
        rocksdb::DB& db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families)
        : db_(db), column_families(column_families) {}

std::unique_ptr<Iterator> RocksDBInvertedList::get_iterator(
        const std::string& start,
        const std::string& end) const {
    return std::make_unique<RocksDBIterator>(RocksDBIterator(
            db_, column_families[kIndexColumnIndex], start, end));
}

void RocksDBInvertedList::add(std::unique_ptr<EncodedDocument> doc) {
    rocksdb::WriteOptions wo;

    // get unique indexes.
    std::unordered_set<idx_t> unique_coarse_idx(
            doc->codes.begin(), doc->codes.end());
    VLOG(100) << "Unique coarse indexes: " << unique_coarse_idx.size();
    // store ivf -> doc mapping.
    for (code_t idx : unique_coarse_idx) {
        Key key = Key{kDefaultTenant, idx, doc->id};
        std::string k_string = key.serialize();

        rocksdb::Status status =
                db_.Put(wo,
                        column_families[kIndexColumnIndex],
                        rocksdb::Slice(k_string),
                        rocksdb::Slice() // store nothing. we only need the key
                                         // to tell us what documents exist.
                );
        assert(status.ok());
        VLOG(100) << "Added document with id: " << doc->id
                  << " to inverted list " << idx;
    }

    // this key is used for all forward indices.
    ForwardIndexKey forward_key = ForwardIndexKey{kDefaultTenant, doc->id};
    auto fks = forward_key.serialize();

    // add document mapping to centroids.
    std::vector<idx_t> unique_coarse_idx_vec(
            unique_coarse_idx.begin(), unique_coarse_idx.end());
    auto mapping_ptr = create_doc_mapping(
            unique_coarse_idx_vec.data(), unique_coarse_idx_vec.size());

    rocksdb::Status mapping_status =
            db_.Put(wo,
                    column_families[kMappingColumnIndex],
                    rocksdb::Slice(fks),
                    rocksdb::Slice(
                            reinterpret_cast<const char*>(
                                    mapping_ptr->GetBufferPointer()),
                            mapping_ptr->GetSize()));
    assert(mapping_status.ok());

    auto doc_ptr = create_inverted_index_document(
            doc->codes.data(), doc->codes.size());
    auto* ptr = doc_ptr->GetBufferPointer();
    auto size = doc_ptr->GetSize();

    // store document codes.

    const rocksdb::Slice slice(reinterpret_cast<const char*>(ptr), size);
    rocksdb::Status code_status = db_.Put(
            wo, column_families[kCodesColumnIndex], rocksdb::Slice(fks), slice);
    assert(code_status.ok());

    assert(doc->residuals.size() > 0);
    VLOG(100) << "Residuals size: " << doc->residuals.size();
    // store document data.
    auto forward_doc_ptr = create_forward_index_document(
            doc->num_tokens,
            doc->doc_id,
            doc->residuals.data(),
            doc->residuals.size());

    auto* forward_ptr = forward_doc_ptr->GetBufferPointer();
    auto forward_size = forward_doc_ptr->GetSize();
    const rocksdb::Slice forward_slice(
            reinterpret_cast<const char*>(forward_ptr), forward_size);

    rocksdb::Status forward_status =
            db_.Put(wo,
                    column_families[kResidualsColumnIndex],
                    rocksdb::Slice(fks),
                    forward_slice);
    assert(forward_status.ok());

    VLOG(100) << "Added document with id: " << doc->id << " to index.";
    ;
};

void RocksDBInvertedList::remove(std::vector<idx_t> ids) {
    for (idx_t id : ids) {
        auto id_map = this->get_mapping(id);
        // delete from the inverse index.
        rocksdb::ReadOptions ro;
        for (auto idx : id_map) {
            Key key = Key{kDefaultTenant, idx, id};
            std::string k_string = key.serialize();
            rocksdb::WriteOptions wo;
            rocksdb::Status status = db_.Delete(
                    wo,
                    column_families[kIndexColumnIndex],
                    rocksdb::Slice(k_string));
            assert(status.ok());
        }

        // delete from all of the forward indices.
        for (size_t i = 1; i < column_families.size(); i++) {
            // ignore the default cf at position 0 since we don't use it.
            auto cf = column_families[i];

            ForwardIndexKey key = ForwardIndexKey{kDefaultTenant, id};
            auto serialized_key = key.serialize();
            rocksdb::WriteOptions wo;
            std::string value;
            rocksdb::Status status = db_.Delete(
                    wo,
                    column_families[kMappingColumnIndex],
                    rocksdb::Slice(serialized_key));
        }
    }
}

std::vector<idx_t> RocksDBInvertedList::get_mapping(idx_t id) const {
    ForwardIndexKey key = ForwardIndexKey{kDefaultTenant, id};
    auto serialized_key = key.serialize();
    rocksdb::ReadOptions ro;
    std::string value;
    rocksdb::Status status =
            db_.Get(ro,
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
        return std::vector<idx_t>();
    }
}

std::vector<std::unique_ptr<EncodedDocument>> RocksDBInvertedList::
        get_residuals(std::vector<idx_t> ids) const {
    std::vector<rocksdb::Slice> keys;
    // rocksdb slices don't take ownership of the underlying data, so we need to
    // keep the strings around.
    std::vector<std::string> key_strings;
    for (idx_t i = 0; i < ids.size(); i++) {
        auto id = ids[i];
        auto key = ForwardIndexKey{kDefaultTenant, id};
        VLOG(100) << "Getting residuals for doc id: " << id;
        auto serialized_key = key.serialize();
        key_strings.push_back(serialized_key);
        keys.push_back(
                rocksdb::Slice(key_strings[i].data(), key_strings[i].size()));
    }

    assert(keys.size() == ids.size());
    VLOG(100) << "Getting num docs: " << keys.size()
              << " from the forward index.";

    std::vector<std::unique_ptr<EncodedDocument>> docs;
    rocksdb::ReadOptions ro;
    std::vector<rocksdb::PinnableSlice> values(keys.size());
    std::vector<rocksdb::Status> statuses(keys.size());

    db_.MultiGet(
            ro,
            column_families[kResidualsColumnIndex],
            keys.size(),
            keys.data(),
            values.data(),
            statuses.data());

    const code_t* empty_codes = nullptr;

    for (size_t i = 0; i < ids.size(); i++) {
        if (statuses[i].ok()) {
            auto doc = GetForwardIndexDocument(values[i].data());
            auto ptr = std::make_unique<EncodedDocument>(EncodedDocument(
                    empty_codes,
                    0,
                    doc->residuals()->data(),
                    doc->residuals()->size(),
                    doc->num_tokens(),
                    ids[i],
                    doc->doc_id()->str()));
            docs.push_back(std::move(ptr));
            // release the memory used by rocksdb for this value.
            values[i].Reset();
        } else {
            LOG(WARNING) << "Could not find document with id: " << ids[i]
                         << " in the forward index.";
            docs.push_back(nullptr);
        }
    }

    return docs;
}

std::vector<std::unique_ptr<EncodedDocument>> RocksDBInvertedList::get_codes(
        std::vector<idx_t> ids) const {
    std::vector<rocksdb::Slice> keys;
    // rocksdb slices don't take ownership of the underlying data, so we need to
    // keep the strings around.
    std::vector<std::string> key_strings;
    for (idx_t i = 0; i < ids.size(); i++) {
        auto id = ids[i];
        auto key = ForwardIndexKey{kDefaultTenant, id};
        VLOG(100) << "Getting codes for doc id: " << id;
        auto serialized_key = key.serialize();
        key_strings.push_back(serialized_key);
        keys.push_back(
                rocksdb::Slice(key_strings[i].data(), key_strings[i].size()));
    }

    assert(keys.size() == ids.size());
    VLOG(100) << "Getting num docs: " << keys.size()
              << " from the forward index.";

    std::vector<std::unique_ptr<EncodedDocument>> docs;
    rocksdb::ReadOptions ro;
    std::vector<rocksdb::PinnableSlice> values(ids.size());
    std::vector<rocksdb::Status> statuses(ids.size());

    db_.MultiGet(
            ro,
            column_families[kCodesColumnIndex],
            keys.size(),
            keys.data(),
            values.data(),
            statuses.data());

    const uint8_t* empty_res = nullptr;

    for (size_t i = 0; i < ids.size(); i++) {
        VLOG(100) << "Status of document with id: " << ids[i]
                  << " is: " << statuses[i].ToString();
        if (statuses[i].ok()) {
            auto doc = GetInvertedIndexDocument(values[i].data());
            auto ptr = std::make_unique<EncodedDocument>(EncodedDocument(
                    doc->codes()->data(),
                    doc->codes()->size(),
                    empty_res,
                    0,
                    0,
                    ids[i],
                    ""));
            docs.push_back(std::move(ptr));
            // release the memory used by rocksdb for this value.
            values[i].Reset();
        } else {
            LOG(WARNING) << "Could not find codes for doc id: " << ids[i];
            docs.push_back(nullptr);
        }
    }

    return docs;
}

void RocksDBInvertedList::delete_entry(idx_t list_no, idx_t id) {
    Key key = Key{kDefaultTenant, list_no, id};
    rocksdb::WriteOptions wo;
    std::string k_string = key.serialize();
    rocksdb::Status status = db_.Delete(wo, rocksdb::Slice(k_string));

    assert(status.ok());
};

} // namespace lintdb