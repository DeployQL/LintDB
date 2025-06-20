#include "lintdb/datasources/RocksDBDataSource.h"
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/status.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>

namespace lintdb {

RocksDBDataSource::RocksDBDataSource(const std::string& path) {
    tenant_id_ = 0;

    rocksdb::Options options;
    options.create_if_missing = false; // We're read-only
    rocksdb::DB* db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, path, &db);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
    }
    db_.reset(db);
}

RocksDBDataSource::~RocksDBDataSource() {
    db_.reset();
}

std::unique_ptr<KeyValueIterator> RocksDBDataSource::scan_prefix(const std::string& prefix) {
    rocksdb::ReadOptions options;
    auto iter = db_->NewIterator(options);
    iter->Seek(prefix);
    return std::make_unique<RocksDBKeyValueIterator>(iter);
}

std::string RocksDBDataSource::get(const std::string& key) {
    std::string value;
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);
    if (status.IsNotFound()) {
        return "";
    }
    if (!status.ok()) {
        throw std::runtime_error("Failed to get value: " + status.ToString());
    }
    return value;
}
} 