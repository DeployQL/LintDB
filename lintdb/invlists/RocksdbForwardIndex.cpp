#include "lintdb/invlists/RocksdbForwardIndex.h"
#include <glog/logging.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>
#include <iostream>
#include <unordered_set>
#include "RocksdbInvertedList.h"
#include "lintdb/assert.h"
#include "lintdb/constants.h"
#include "lintdb/exception.h"
#include "lintdb/schema/forward_index_generated.h"
#include "lintdb/schema/util.h"
#include "lintdb/invlists/ForwardIndexIterator.h"

namespace lintdb {

RocksdbForwardIndex::RocksdbForwardIndex(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
        const Version& version)
        : version(version), db_(db), column_families(column_families) {}

void RocksdbForwardIndex::remove(
        const uint64_t tenant,
        std::vector<idx_t> ids) {
    for (idx_t id : ids) {
        // it's easier to skip the inverted index column families.
        if (id == kIndexColumnIndex || id == kMappingColumnIndex) {
            continue;
        }
        // delete from all of the forward indices.
        for (size_t i = 1; i < column_families.size(); i++) {
            // ignore the default cf at position 0 since we don't use it.
            ForwardIndexKey key = ForwardIndexKey{tenant, id};
            auto serialized_key = key.serialize();
            rocksdb::WriteOptions wo;
            std::string value;
            rocksdb::Status status = db_->Delete(
                    wo, column_families[i], rocksdb::Slice(serialized_key));
        }
    }
}

std::vector<std::unique_ptr<DocumentResiduals>> RocksdbForwardIndex::
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

std::vector<std::unique_ptr<DocumentCodes>> RocksdbForwardIndex::get_codes(
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

std::vector<std::unique_ptr<DocumentMetadata>> RocksdbForwardIndex::
        get_metadata(const uint64_t tenant, const std::vector<idx_t>& ids)
                const {
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

//#pragma omp parallel for
    for (int i = 0; i < key_strings.size(); i++) {
        auto key = rocksdb::Slice(key_strings[i].data(), key_strings[i].size());

        statuses[i] =
                db_->Get(ro, column_families[kDocColumnIndex], key, &values[i]);
    }

    for (size_t i = 0; i < ids.size(); i++) {
        if (statuses[i].ok() && values[i].size() > 0) {
            auto doc = values[i].ToString();
            std::unique_ptr<DocumentMetadata> metadata =
                    DocumentMetadata::deserialize(doc);

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

void RocksdbForwardIndex::merge(
        rocksdb::DB* db,
        std::vector<rocksdb::ColumnFamilyHandle*>& cfs) {
    // very weak check to make sure the column families are the same.
    LINTDB_THROW_IF_NOT(cfs.size() == column_families.size());

#pragma omp for
    for (size_t i = 1; i < cfs.size(); i++) {
        // it's easier to skip the inverted index column families.
        // the forward index uses the rest.
        if (i == kIndexColumnIndex || i == kMappingColumnIndex) {
            continue;
        }
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

std::unique_ptr<ForwardIndexIterator> RocksdbForwardIndex::get_iterator(
            const uint64_t tenant,
            idx_t column_index) const {
    return std::make_unique<ForwardIndexIterator>(ForwardIndexIterator(
            db_, column_families[kIndexColumnIndex], tenant));
}

} // namespace lintdb