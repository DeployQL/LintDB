#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lintdb/datasources/DataSource.h"
#include "rocksdb/db.h"

namespace lintdb {

/**
 * RocksDBKeyValueIterator implements the KeyValueIterator interface using RocksDB.
 */
class RocksDBKeyValueIterator : public KeyValueIterator {
public:
    explicit RocksDBKeyValueIterator(rocksdb::Iterator* iter) : iter_(iter) {}

    bool is_valid() const override {
        return iter_->Valid();
    }

    void advance() override {
        iter_->Next();
    }

    std::string key() const override {
        return iter_->key().ToString();
    }

    std::string value() const override {
        return iter_->value().ToString();
    }

private:
    std::unique_ptr<rocksdb::Iterator> iter_;
};

/**
 * RocksDBDataSource implements the DataSource interface using RocksDB.
 */
class RocksDBDataSource : public DataSource {
public:
    /**
     * Create a new RocksDB data source.
     * @param path The path to the RocksDB database
     * @param schema The schema of the data
     */
    RocksDBDataSource(const std::string& path);

    ~RocksDBDataSource() override;

    Schema schema() const override {
        return schema_;
    }

    std::unique_ptr<KeyValueIterator> scan_prefix(const std::string& prefix) override;
    std::string get(const std::string& key) override;

private:
    std::unique_ptr<rocksdb::DB> db_;
    Schema schema_;
    uint64_t tenant_id_;
}; 
}