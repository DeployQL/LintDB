#include <gtest/gtest.h>
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/cf.h"
#include "lintdb/index.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/RocksdbInvertedListV2.h"
#include "lintdb/version.h"
#include "util.h"

using ::testing::TestWithParam;
using ::testing::Values;

class InvertedListTest : public TestWithParam<lintdb::IndexEncoding> {
   public:
    ~InvertedListTest() override {}
    void SetUp() override {
        version = lintdb::Version();
        temp_db = create_temporary_directory();
        rocksdb::Options options;
        options.create_if_missing = true;
        options.create_missing_column_families = true;

        auto cfs = lintdb::create_column_families();

        rocksdb::DB* ptr;
        rocksdb::Status s = rocksdb::DB::Open(
                options, temp_db, cfs, &column_families, &ptr);

        assert(s.ok());
        this->db = std::shared_ptr<rocksdb::DB>(ptr);
    }
    void TearDown() override {
        for (auto cf : column_families) {
            db->DestroyColumnFamilyHandle(cf);
        }
        std::filesystem::remove_all(temp_db);
    }

   protected:
    lintdb::Version version;
    std::filesystem::path temp_db;
    std::shared_ptr<rocksdb::DB> db;
    std::vector<rocksdb::ColumnFamilyHandle*> column_families;
};

TEST_F(InvertedListTest, StoresCodesCorrectly) {
    lintdb::RocksdbInvertedListV2 invlist(db, column_families, version);

    lintdb::EncodedDocument doc(
            {1, 2, 3}, std::vector<residual_t>(3 * 1, 1), 3, 555, 1, {});

    invlist.add(0, &doc);

    lintdb::Key start{0, 1, 0, true};
    lintdb::Key end{0, 1, std::numeric_limits<idx_t>::max(), false};
    auto options = rocksdb::ReadOptions();
    rocksdb::Slice end_slice(end.serialize());
    options.iterate_upper_bound = &end_slice;
    std::string start_string = start.serialize();
    auto it = std::unique_ptr<rocksdb::Iterator>(
            db->NewIterator(options, column_families[1]));

    rocksdb::Slice prefix(start_string);
    it->Seek(prefix);
    int count = 0;
    for (; it->Valid(); it->Next()) {
        auto k = it->key().ToString();
        auto key = lintdb::TokenKey::from_slice(k);

        auto id = key.doc_id;
        EXPECT_EQ(id, idx_t(555));

        auto val = it->value().ToString();
        ASSERT_EQ(val.size(), 1);
        count++;
    }
    ASSERT_EQ(count, 3);

    auto it1 = invlist.get_iterator(0, 1);
    for (; it1->has_next(); it1->next()) {
        auto key = it1->get_key();
        ASSERT_EQ(key.doc_id, 555);
        ASSERT_EQ(key.token_id, 0);

        auto val = it1->get_value();
        ASSERT_EQ(val.partial_residuals.size(), 1);
    }

    auto it2 = invlist.get_iterator(0, 2);
    for (; it2->has_next(); it2->next()) {
        auto key = it2->get_key();
        ASSERT_EQ(key.doc_id, 555);
        ASSERT_EQ(key.token_id, 1);

        auto val = it2->get_value();
        ASSERT_EQ(val.partial_residuals.size(), 1);
    }

    auto it3 = invlist.get_iterator(0, 3);
    for (; it3->has_next(); it3->next()) {
        auto key = it3->get_key();
        ASSERT_EQ(key.doc_id, 555);
        ASSERT_EQ(key.token_id, 2);

        auto val = it3->get_value();
        ASSERT_EQ(val.partial_residuals.size(), 1);
    }
}