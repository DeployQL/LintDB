#pragma once

#include <rocksdb/db.h>
#include <rocksdb/iterator.h>
#include "lintdb/invlists/PostingData.h"
#include "lintdb/version.h"
#include <vector>

namespace lintdb {

    class IndexWriter {
    private:
        std::shared_ptr<rocksdb::DB> db;
        std::vector<rocksdb::ColumnFamilyHandle*>& column_families;
        const Version& version;

    public:
        IndexWriter(std::shared_ptr<rocksdb::DB> db,
                    std::vector<rocksdb::ColumnFamilyHandle*>& column_families,
                    const Version& version
        );

        void write(const BatchPostingData& batch_posting_data);
    };

} // lintdb
