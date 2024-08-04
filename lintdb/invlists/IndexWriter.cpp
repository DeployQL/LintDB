#include "IndexWriter.h"
#include <glog/logging.h>
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include "lintdb/api.h"
#include "lintdb/assert.h"
#include "lintdb/constants.h"
#include "lintdb/invlists/PostingData.h"

namespace lintdb {
IndexWriter::IndexWriter(
        std::shared_ptr<rocksdb::DB> db,
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
        const Version& version)
        : db(db), column_families(column_families), version(version) {}

/**
 * Write will batch write all document data to the database.
 * @param batch_posting_data
 */
void IndexWriter::write(const BatchPostingData& batch_posting_data) {
    rocksdb::WriteBatch batch;

    // write all inverted index data
    for (const auto& posting : batch_posting_data.inverted) {
        batch.Put(
                column_families[kIndexColumnIndex],
                rocksdb::Slice(posting.key),
                rocksdb::Slice(posting.value));
    }

    // write all mappings
    for (const auto& posting : batch_posting_data.inverted_mapping) {
        batch.Put(
                column_families[kMappingColumnIndex],
                rocksdb::Slice(posting.key),
                rocksdb::Slice(posting.value));
    }

    // write all document data
    batch.Put(
            column_families[kDocColumnIndex],
            rocksdb::Slice(batch_posting_data.forward.key),
            rocksdb::Slice(batch_posting_data.forward.value));

    // write all context data
    for (const auto& posting : batch_posting_data.context) {
        batch.Put(
                column_families[kCodesColumnIndex],
                rocksdb::Slice(posting.key),
                rocksdb::Slice(posting.value));
    }

    auto status = db->Write(rocksdb::WriteOptions(), &batch);
    assert(status.ok());

    LINTDB_THROW_IF_NOT(status.ok());
}
} // namespace lintdb