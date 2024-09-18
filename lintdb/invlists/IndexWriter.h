#pragma once

#include <vector>
#include <memory>
#include "lintdb/invlists/PostingData.h"
#include "lintdb/version.h"

namespace rocksdb {
    class DB;
    class ColumnFamilyHandle;
}

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
