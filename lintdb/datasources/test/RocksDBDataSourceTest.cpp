#include <gtest/gtest.h>
#include <filesystem>
#include "lintdb/datasources/RocksDBDataSource.h"
#include "lintdb/query/Schema.h"

using namespace lintdb;

class RocksDBDataSourceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for the test database
        db_path_ = std::filesystem::temp_directory_path() / "lintdb_test";
        std::filesystem::create_directories(db_path_);

        // Create a simple schema
        Schema schema;
        schema.add_field(Field(DataType::FLOAT, "vector"));
        schema.add_field(Field(DataType::STRING, "text"));

        // Create the data source
        data_source_ = std::make_unique<RocksDBDataSource>(db_path_.string(), schema);
    }

    void TearDown() override {
        data_source_.reset();
        std::filesystem::remove_all(db_path_);
    }

    std::filesystem::path db_path_;
    std::unique_ptr<RocksDBDataSource> data_source_;
};

TEST_F(RocksDBDataSourceTest, GetNonexistent) {
    auto retrieved = data_source_->get("nonexistent_key");
    EXPECT_TRUE(retrieved.empty());
}

TEST_F(RocksDBDataSourceTest, ScanEmpty) {
    auto iter = data_source_->scan();
    EXPECT_FALSE(iter->is_valid());
}

TEST_F(RocksDBDataSourceTest, ScanPrefixEmpty) {
    auto iter = data_source_->scan_prefix("prefix");
    EXPECT_FALSE(iter->is_valid());
} 