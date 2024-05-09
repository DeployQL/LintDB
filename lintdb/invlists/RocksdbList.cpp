#include "lintdb/invlists/RocksdbList.h"
#include <glog/logging.h>
#include <iostream>
#include <unordered_set>
#include "lintdb/assert.h"
#include "lintdb/constants.h"
#include "lintdb/exception.h"
#include "lintdb/schema/forward_index_generated.h"
#include "lintdb/schema/util.h"
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>

namespace lintdb {
RocksDBIterator::RocksDBIterator(
        std::shared_ptr<rocksdb::DB> db,
        rocksdb::ColumnFamilyHandle* column_family,
        const uint64_t tenant,
        const idx_t inverted_list)
        : Iterator(), tenant(tenant), inverted_index(inverted_list) {
    cf = column_family->GetID();
    prefix = Key{tenant, inverted_list, 0, true}.serialize();
    // end_key = Key{tenant, inverted_list, std::numeric_limits<idx_t>::max(),
    // true}.serialize();

    prefix_slice = rocksdb::Slice(this->prefix);
    // upper_bound =
    // std::make_unique<rocksdb::Slice>(rocksdb::Slice(this->end_key));

    auto options = rocksdb::ReadOptions();
    // options.iterate_upper_bound = upper_bound.get();

    this->it = std::unique_ptr<rocksdb::Iterator>(
            db->NewIterator(options, column_family));
    it->Seek(this->prefix);
}

template <typename DBType>
RocksDBInvertedList<DBType>::RocksDBInvertedList(
        std::shared_ptr<DBType> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families)
        : db_(db), column_families(column_families) {}

template <typename DBType>
std::unique_ptr<Iterator> RocksDBInvertedList<DBType>::get_iterator(
        const uint64_t tenant,
        const idx_t inverted_list) const {
    return std::make_unique<RocksDBIterator>(RocksDBIterator(
            db_, column_families[kIndexColumnIndex], tenant, inverted_list));
}

template <typename DBType>
void RocksDBInvertedList<DBType>::add(
        const uint64_t tenant,
        std::unique_ptr<EncodedDocument> doc) {
    rocksdb::WriteOptions wo;
    // get unique indexes.
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

    // this key is used for all forward indices.
    ForwardIndexKey forward_key = ForwardIndexKey{tenant, doc->id};
    auto fks = forward_key.serialize();

    // add document mapping to centroids.
    std::vector<idx_t> unique_coarse_idx_vec(
            unique_coarse_idx.begin(), unique_coarse_idx.end());
    auto mapping_ptr = create_doc_mapping(
            unique_coarse_idx_vec.data(), unique_coarse_idx_vec.size());

    rocksdb::Status mapping_status = db_->Put(
            wo,
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
    rocksdb::Status code_status = db_->Put(
            wo, column_families[kCodesColumnIndex], rocksdb::Slice(fks), slice);
    assert(code_status.ok());

    assert(doc->residuals.size() > 0);
    VLOG(100) << "Residuals size: " << doc->residuals.size();
    // store document residual data.
    auto forward_doc_ptr = create_forward_index_document(
            doc->num_tokens, doc->residuals.data(), doc->residuals.size());

    auto* forward_ptr = forward_doc_ptr->GetBufferPointer();
    auto forward_size = forward_doc_ptr->GetSize();
    const rocksdb::Slice forward_slice(
            reinterpret_cast<const char*>(forward_ptr), forward_size);

    rocksdb::Status forward_status = db_->Put(
            wo,
            column_families[kResidualsColumnIndex],
            rocksdb::Slice(fks),
            forward_slice);
    assert(forward_status.ok());

    VLOG(100) << "Added document with id: " << doc->id << " to index.";
};

template <typename DBType>
void RocksDBInvertedList<DBType>::remove(
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

        // delete from all of the forward indices.
        for (size_t i = 1; i < column_families.size(); i++) {
            // ignore the default cf at position 0 since we don't use it.
            ForwardIndexKey key = ForwardIndexKey{tenant, id};
            auto serialized_key = key.serialize();
            rocksdb::WriteOptions wo;
            std::string value;
            rocksdb::Status status = db_->Delete(
                    wo,
                    column_families[i],
                    rocksdb::Slice(serialized_key));
        }
    }
}

template <typename DBType>
std::vector<idx_t> RocksDBInvertedList<DBType>::get_mapping(
        const uint64_t tenant,
        idx_t id) const {
    ForwardIndexKey key = ForwardIndexKey{tenant, id};
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
        return std::vector<idx_t>();
    }
}

template <typename DBType>
std::vector<std::unique_ptr<DocumentResiduals>> RocksDBInvertedList<DBType>::
        get_residuals(const uint64_t tenant, const std::vector<idx_t>& ids)
                const {
    std::vector<std::string> key_strings;
    std::vector<rocksdb::Slice> keys;
    for (idx_t i = 0; i < ids.size(); i++) {
        auto id = ids[i];
        key_strings.push_back(ForwardIndexKey{tenant, id}.serialize());
        keys.push_back(rocksdb::Slice(key_strings[i]));
    }

    assert(key_strings.size() == ids.size());
    VLOG(100) << "Getting num docs: " << key_strings.size()
              << " from the forward index.";

    std::vector<std::unique_ptr<DocumentResiduals>> docs;
    rocksdb::ReadOptions ro;
    std::vector<rocksdb::PinnableSlice> values(key_strings.size());
    std::vector<rocksdb::Status> statuses(key_strings.size());

    // see get_codes() for why this method isn't working.
    // db_->MultiGet(
    //         ro,
    //         column_families[kResidualsColumnIndex],
    //         keys.size(),
    //         keys.data(),
    //         values.data(),
    //         statuses.data());

#pragma omp parallel for
    for (int i = 0; i < key_strings.size(); i++) {
        auto key = rocksdb::Slice(key_strings[i]);

        statuses[i] = db_->Get(
                ro, column_families[kResidualsColumnIndex], key, &values[i]);
    }

    for (size_t i = 0; i < ids.size(); i++) {
        if (statuses[i].ok()) {
            auto doc = GetForwardIndexDocument(values[i].data());
            auto ptr = std::make_unique<DocumentResiduals>(DocumentResiduals(
                    ids[i],
                    doc->residuals()->data(),
                    doc->residuals()->size(),
                    doc->num_tokens()));
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

template <typename DBType>
std::vector<std::unique_ptr<DocumentCodes>> RocksDBInvertedList<DBType>::
        get_codes(const uint64_t tenant, const std::vector<idx_t>& ids) const {
    std::vector<std::string> key_strings;
    std::vector<rocksdb::Slice> keys;
    // rocksdb slices don't take ownership of the underlying data, so we need to
    // keep the strings around.
    for (idx_t i = 0; i < ids.size(); i++) {
        auto id = ids[i];
        key_strings.push_back(ForwardIndexKey{tenant, id}.serialize());
        keys.push_back(rocksdb::Slice(key_strings[i]));
    }

    assert(key_strings.size() == ids.size());
    VLOG(100) << "Getting num docs: " << key_strings.size()
              << " from the forward index.";

    std::vector<std::unique_ptr<DocumentCodes>> docs;
    rocksdb::ReadOptions ro;
    std::vector<rocksdb::PinnableSlice> values(ids.size());
    std::vector<rocksdb::Status> statuses(ids.size());

    // NOTE (MB): for some reason, multiget doesn't work. The issue is that
    // the rocksdb::Slices don't seem to retain the string information
    // **before** we make this call. I can't figure out why this is happening
    // here, but not with individual get() calls. Multiget also works fine in an
    // isolated binary, but copying that code here results in failure as well.
    // I'm guessing there is an issue with this invlist accessing the memory of
    // the strings for some reason. db_->MultiGet(
    //         ro,
    //         column_families[kCodesColumnIndex],
    //         keys.size(),
    //         keys.data(),
    //         values.data(),
    //         statuses.data());
#pragma omp parallel for
    for (int i = 0; i < key_strings.size(); i++) {
        auto key = rocksdb::Slice(key_strings[i].data(), key_strings[i].size());

        statuses[i] = db_->Get(
                ro, column_families[kCodesColumnIndex], key, &values[i]);
    }

    for (size_t i = 0; i < ids.size(); i++) {
        if (statuses[i].ok()) {
            auto doc = GetInvertedIndexDocument(values[i].data());
            auto ptr = std::make_unique<DocumentCodes>(DocumentCodes(
                    ids[i],
                    doc->codes()->data(),
                    doc->codes()->size(),
                    doc->codes()->size()));
            docs.push_back(std::move(ptr));
            // release the memory used by rocksdb for this value.
            values[i].Reset();
        } else {
            LOG(ERROR) << "Could not find codes for doc id: " << ids[i];
            LOG(ERROR) << "rocksdb: " << statuses[i].ToString();
            docs.push_back(nullptr);
        }
    }

    return docs;
}

template <typename DBType>
std::vector<std::unique_ptr<DocumentMetadata>> RocksDBInvertedList<DBType>::get_metadata(
    const uint64_t tenant,
    const std::vector<idx_t>& ids) const {

    std::vector<std::string> key_strings;
    std::vector<rocksdb::Slice> keys;
    // rocksdb slices don't take ownership of the underlying data, so we need to
    // keep the strings around.
    for (idx_t i = 0; i < ids.size(); i++) {
        auto id = ids[i];
        key_strings.push_back(ForwardIndexKey{tenant, id}.serialize());
        keys.push_back(rocksdb::Slice(key_strings[i]));
    }

    assert(key_strings.size() == ids.size());
    VLOG(100) << "Getting num docs: " << key_strings.size()
              << " from the metadata index.";

    std::vector<std::unique_ptr<DocumentMetadata>> docs;
    rocksdb::ReadOptions ro;
    std::vector<rocksdb::PinnableSlice> values(ids.size());
    std::vector<rocksdb::Status> statuses(ids.size());

    #pragma omp parallel for
    for (int i = 0; i < key_strings.size(); i++) {
        auto key = rocksdb::Slice(key_strings[i].data(), key_strings[i].size());

        statuses[i] = db_->Get(
                ro, column_families[kDocColumnIndex], key, &values[i]);
    }

    for (size_t i = 0; i < ids.size(); i++) {
        if (statuses[i].ok() && values[i].size() > 0) {
            // auto doc = values[i].data();
            auto doc = values[i].ToString();
            std::unique_ptr<DocumentMetadata> metadata = DocumentMetadata::deserialize(doc);

            for(auto& [key, value]: metadata->metadata) {
                VLOG(100) << "Metadata: " << key << " -> " << value;
            }

            docs.push_back(std::move(metadata));
            // release the memory used by rocksdb for this value.
            values[i].Reset();
        } else {
            LOG(ERROR) << "Could not find metadata for doc id: " << ids[i];
            LOG(ERROR) << "rocksdb: " << statuses[i].ToString();
            docs.push_back(nullptr);
        }
    }

    return docs;
}

template <typename DBType>
void RocksDBInvertedList<DBType>::delete_entry(
        idx_t list_no,
        const uint64_t tenant,
        idx_t id) {
    Key key = Key{tenant, list_no, id};
    rocksdb::WriteOptions wo;
    std::string k_string = key.serialize();
    rocksdb::Status status = db_->Delete(wo, rocksdb::Slice(k_string));

    assert(status.ok());
};

template <typename DBType>
void RocksDBInvertedList<DBType>::merge(
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

WritableRocksDBInvertedList::WritableRocksDBInvertedList(
        std::shared_ptr<rocksdb::OptimisticTransactionDB> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families)
        : RocksDBInvertedList<rocksdb::OptimisticTransactionDB>(
                  db,
                  column_families) {}

void WritableRocksDBInvertedList::add(
        const uint64_t tenant,
        std::unique_ptr<EncodedDocument> doc) {
    rocksdb::WriteOptions wo;
    // get unique indexes.
    std::unordered_set<idx_t> unique_coarse_idx(
            doc->codes.begin(), doc->codes.end());
    VLOG(100) << "Unique coarse indexes: " << unique_coarse_idx.size();

    std::unique_ptr<rocksdb::Transaction> txn =
            std::unique_ptr<rocksdb::Transaction>(db_->BeginTransaction(wo));

    rocksdb::WriteBatch batch;

    // store inverted index.
    for(code_t idx : unique_coarse_idx) {
        VLOG(100) << "Adding document with id: " << doc->id
                      << " to inverted list " << idx;
        auto index_status = batch.Put(
            column_families[kIndexColumnIndex],
            rocksdb::Slice(Key{tenant, idx, doc->id}.serialize()),
            rocksdb::Slice()
        );
        LINTDB_THROW_IF_NOT(index_status.ok());
    }

    // store forward doc id -> coarse idx mapping.
    ForwardIndexKey forward_key = ForwardIndexKey{tenant, doc->id};
    auto fks = forward_key.serialize();
    std::vector<idx_t> unique_coarse_idx_vec(
                unique_coarse_idx.begin(), unique_coarse_idx.end());
    auto mapping_ptr = create_doc_mapping(
            unique_coarse_idx_vec.data(), unique_coarse_idx_vec.size());
    auto mapping_status = batch.Put(
        column_families[kMappingColumnIndex],
        rocksdb::Slice(fks),
        rocksdb::Slice(
                reinterpret_cast<const char*>(
                        mapping_ptr->GetBufferPointer()),
                mapping_ptr->GetSize())
    );
    LINTDB_THROW_IF_NOT(mapping_status.ok());

    // store document codes.
    auto doc_ptr = create_inverted_index_document(
            doc->codes.data(), doc->codes.size());
    auto* ptr = doc_ptr->GetBufferPointer();
    auto size = doc_ptr->GetSize();

    const rocksdb::Slice slice(reinterpret_cast<const char*>(ptr), size);
    auto codes_status = batch.Put(column_families[kCodesColumnIndex], rocksdb::Slice(fks), slice);
    LINTDB_THROW_IF_NOT(codes_status.ok());

    // store document residuals.
    auto forward_doc_ptr = create_forward_index_document(
        doc->num_tokens, doc->residuals.data(), doc->residuals.size());

    auto* forward_ptr = forward_doc_ptr->GetBufferPointer();
    auto forward_size = forward_doc_ptr->GetSize();
    const rocksdb::Slice forward_slice(
            reinterpret_cast<const char*>(forward_ptr), forward_size);

    rocksdb::Status forward_status = batch.Put(
            column_families[kResidualsColumnIndex],
            rocksdb::Slice(fks),
            forward_slice);
    LINTDB_THROW_IF_NOT(forward_status.ok());

    // store document metadata.
    std::string metadata_serialized = doc->serialize_metadata();

    const rocksdb::Slice metadata_slice(metadata_serialized);
    rocksdb::Status metadata_status = batch.Put(
            column_families[kDocColumnIndex],
            rocksdb::Slice(fks),
            metadata_slice);

    auto status = db_->Write(wo, &batch);
    assert(status.ok());

    LINTDB_THROW_IF_NOT(status.ok());
};

ReadOnlyRocksDBInvertedList::ReadOnlyRocksDBInvertedList(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families)
        : RocksDBInvertedList<rocksdb::DB>(db, column_families) {}

void ReadOnlyRocksDBInvertedList::add(
        const uint64_t tenant,
        std::unique_ptr<EncodedDocument> docs) {
    // throw an error.
    throw LintDBException("Cannot add to a read-only index.");
}
void ReadOnlyRocksDBInvertedList::remove(
        const uint64_t tenant,
        std::vector<idx_t> ids) {
    // throw an error.
    throw LintDBException("Cannot remove from a read-only index.");
}

void ReadOnlyRocksDBInvertedList::merge(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*> cfs) {
    // throw an error.
    throw LintDBException("Cannot merge a read-only index.");
}

void ReadOnlyRocksDBInvertedList::delete_entry(
        idx_t list_no,
        const uint64_t tenant,
        idx_t id) {
    // throw an error.
    throw LintDBException("Cannot delete from a read-only index.");
}

} // namespace lintdb