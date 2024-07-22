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
#include "lintdb/invlists/KeyBuilder.h"

using ::testing::Test;
using ::testing::Values;

class InvertedListTest : public Test {
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


    auto one = lintdb::create_index_id(0, 1, lintdb::DataType::QUANTIZED_TENSOR, 1, 555);
    auto two = lintdb::create_index_id(0, 1, lintdb::DataType::QUANTIZED_TENSOR, 1, 556);
    auto three = lintdb::create_index_id(0, 1, lintdb::DataType::QUANTIZED_TENSOR, 3, 555);
    rocksdb::WriteOptions wo;
    this->db->Put(wo, column_families[lintdb::kIndexColumnIndex], one, "value");
    this->db->Put(wo, column_families[lintdb::kIndexColumnIndex], two, "value");
    this->db->Put(wo, column_families[lintdb::kIndexColumnIndex], three, "value");

    std::string prefix = lintdb::create_index_prefix(0, 1, lintdb::DataType::QUANTIZED_TENSOR, 1);
    auto it1 = invlist.get_iterator(prefix);

    // inverted list should have 2 entries
    EXPECT_TRUE(it1->is_valid());
    auto key = it1->get_key();
    ASSERT_EQ(key.doc_id(), 555);

    std::string val = it1->get_value();
    ASSERT_EQ(val, "value");

    it1->next();

    EXPECT_TRUE(it1->is_valid());
    key = it1->get_key();
    ASSERT_EQ(key.doc_id(), 556);

    val = it1->get_value();
    ASSERT_EQ(val, "value");

    // only two documents.
    it1->next();
    EXPECT_FALSE(it1->is_valid());


    std::string prefix_three = lintdb::create_index_prefix(0, 1, lintdb::DataType::QUANTIZED_TENSOR, 3);
    auto it3 = invlist.get_iterator(prefix_three);

    EXPECT_TRUE(it3->is_valid());

    auto key_three = it3->get_key();
    ASSERT_EQ(key_three.doc_id(), 555);

    std::string val_three = it3->get_value();
    ASSERT_EQ(val_three, "value");

    // only one document.
    it3->next();
    EXPECT_FALSE(it3->is_valid());

}