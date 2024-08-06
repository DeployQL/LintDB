#pragma once

#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include <vector>
#include "lintdb/invlists/PostingData.h"
#include "lintdb/version.h"

namespace lintdb {

class IIndexWriter {
   public:
    virtual void write(const BatchPostingData& batch_posting_data) = 0;

    virtual ~IIndexWriter() = default;
};

class IndexWriter : public IIndexWriter {
   private:
    std::shared_ptr<rocksdb::DB> db;
    std::vector<rocksdb::ColumnFamilyHandle*>& column_families;
    const Version& version;

   public:
    IndexWriter(
            std::shared_ptr<rocksdb::DB> db,
            std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
            const Version& version);

    void write(const BatchPostingData& batch_posting_data) override;
};

} // namespace lintdb