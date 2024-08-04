#ifndef LINTDB_INVLISTS_INVERTED_LIST_H
#define LINTDB_INVLISTS_INVERTED_LIST_H

#include <stddef.h>
#include <memory>
#include <vector>
#include "lintdb/api.h"
#include "lintdb/constants.h"
#include "lintdb/invlists/ContextIterator.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/ForwardIndexIterator.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/schema/Schema.h"

namespace lintdb {
/**
 * InvertedList manages the storage of centroid -> codes mappping.
 *
 * InvertedLists are expected to be fast. The more data stored in the
 * invertedList, the slower it will become.
 *
 * We also expect the InvertedList to manage a mapping of document -> centroids
 * to facilitate deletion.
 */
struct InvertedList {
    virtual void remove(
            const uint64_t tenant,
            std::vector<idx_t> ids,
            const uint8_t field,
            const DataType data_type,
            const std::vector<FieldType> field_types) = 0;
    virtual void merge(
            rocksdb::DB* db,
            std::vector<rocksdb::ColumnFamilyHandle*>& cfs) = 0;

    virtual std::unique_ptr<Iterator> get_iterator(
            const std::string& prefix) const = 0;

    virtual std::unique_ptr<ContextIterator> get_context_iterator(
            const uint64_t tenant,
            const uint8_t field_id) const = 0;

    virtual std::vector<idx_t> get_mapping(const uint64_t tenant, idx_t id)
            const = 0;

    virtual ~InvertedList() = default;
};

/**
 * ForwardIndex helps retrieve document data from the index.
 */
struct ForwardIndex {
    virtual std::vector<std::map<uint8_t, SupportedTypes>> get_metadata(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const = 0;

    virtual void remove(const uint64_t tenant, std::vector<idx_t> ids) = 0;

    virtual void merge(
            rocksdb::DB* db,
            std::vector<rocksdb::ColumnFamilyHandle*>& cfs) = 0;

    virtual std::unique_ptr<ForwardIndexIterator> get_iterator(
            const uint64_t tenant,
            const idx_t inverted_list) const = 0;

    virtual ~ForwardIndex() = default;
};
} // namespace lintdb

#endif