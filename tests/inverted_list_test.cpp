#include <gtest/gtest.h>
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include "lintdb/cf.h"
#include "lintdb/index.h"
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
    lintdb::RocksdbInvertedList invlist(db, column_families, version);


    lintdb::TokenKey one{0, 1, 1, 0, 0};
    lintdb::TokenKey two{0, 1, 2, 0, 0};
    lintdb::TokenKey three{0, 1, 3, 0, 0};
    rocksdb::WriteOptions wo;
    this->db->Put(wo, column_families[lintdb::kCodesColumnIndex], one.serialize(), "value");
    this->db->Put(wo, column_families[lintdb::kCodesColumnIndex], two.serialize(), "value");
    this->db->Put(wo, column_families[lintdb::kCodesColumnIndex], three.serialize(), "value");


    auto it1 = invlist.get_iterator(0, 1, 1);
    for (; it1->has_next(); it1->next()) {
        auto key = it1->get_key();
        ASSERT_EQ(key.doc_id, 555);
        ASSERT_EQ(key.token_id, 0);

        std::string val = it1->get_value();
        ASSERT_EQ(val, "value");
    }

    auto it2 = invlist.get_iterator(0, 1, 2);
    for (; it2->has_next(); it2->next()) {
        auto key = it2->get_key();
        ASSERT_EQ(key.doc_id, 555);
        ASSERT_EQ(key.token_id, 1);

        std::string val = it1->get_value();
        ASSERT_EQ(val, "value");
    }

    auto it3 = invlist.get_iterator(0, 1, 3);
    for (; it3->has_next(); it3->next()) {
        auto key = it3->get_key();
        ASSERT_EQ(key.doc_id, 555);
        ASSERT_EQ(key.token_id, 2);

        std::string val = it1->get_value();
        ASSERT_EQ(val, "value");
    }
}