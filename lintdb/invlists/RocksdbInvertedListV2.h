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

        void add(uint64_t tenant, EncodedDocument* doc) override;
    };
}