#ifndef LINTDB_CF_H
#define LINTDB_CF_H

#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include "lintdb/constants.h"

namespace lintdb {
namespace {
rocksdb::ColumnFamilyOptions create_index_table_options() {
    rocksdb::ColumnFamilyOptions index_options;
    rocksdb::BlockBasedTableOptions table_options;
    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
    index_options.table_factory.reset(
            rocksdb::NewBlockBasedTableFactory(table_options));

    // the inverted index uses 8 bytes for the tenant, and 4 bytes for the
    // inverted list id.
    index_options.prefix_extractor.reset(rocksdb::NewCappedPrefixTransform(12));

    return index_options;
};
} // namespace
inline std::vector<rocksdb::ColumnFamilyDescriptor> create_column_families() {
    return {rocksdb::ColumnFamilyDescriptor(
                    rocksdb::kDefaultColumnFamilyName,
                    rocksdb::ColumnFamilyOptions()),
            rocksdb::ColumnFamilyDescriptor(
                    kIndexColumnFamily, create_index_table_options()),
            rocksdb::ColumnFamilyDescriptor(
                    kForwardColumnFamily, rocksdb::ColumnFamilyOptions()),
            rocksdb::ColumnFamilyDescriptor(
                    kCodesColumnFamily, rocksdb::ColumnFamilyOptions()),
            rocksdb::ColumnFamilyDescriptor(
                    kResidualsColumnFamily, rocksdb::ColumnFamilyOptions()),
            rocksdb::ColumnFamilyDescriptor(
                    kMappingColumnFamily, rocksdb::ColumnFamilyOptions()),
            rocksdb::ColumnFamilyDescriptor(
                    kDocColumnFamily, rocksdb::ColumnFamilyOptions())};
}

} // namespace lintdb

#endif