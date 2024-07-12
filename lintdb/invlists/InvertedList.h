#ifndef LINTDB_INVLISTS_INVERTED_LIST_H
#define LINTDB_INVLISTS_INVERTED_LIST_H

#include <stddef.h>
#include <memory>
#include <vector>
#include "lintdb/api.h"
#include "lintdb/constants.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/invlists/ForwardIndexIterator.h"
#include "lintdb/invlists/ContextIterator.h"

namespace lintdb {
/**
 * InvertedList manages the storage of centroid -> codes mappping.
 *
 * InvertedLists are expected to be fast. The more data stored in the invertedList,
 * the slower it will become.
 *
 * We also expect the InvertedList to manage a mapping of document -> centroids to facilitate
 * deletion.
 */
struct InvertedList {
    virtual void remove(const uint64_t tenant, std::vector<idx_t> ids) = 0;
    virtual void merge(
            rocksdb::DB* db,
            std::vector<rocksdb::ColumnFamilyHandle*>& cfs) = 0;

    virtual std::unique_ptr<Iterator> get_iterator(
            const uint64_t tenant,
            const uint8_t field_id,
            const idx_t inverted_list) const = 0;

    virtual std::unique_ptr<ContextIterator> get_context_iterator(
            const uint64_t tenant,
            const uint8_t field_id
            ) const = 0;

    virtual std::vector<idx_t> get_mapping(const uint64_t tenant, idx_t id)
            const = 0;

    virtual ~InvertedList() = default;
};


/**
 * ForwardIndex helps retrieve document data from the index.
 */
struct ForwardIndex {
    /**
     * get retrieves from the forward index. This could be any data that is
     * mapped by a document id.
     */
    virtual std::vector<std::unique_ptr<DocumentCodes>> get_codes(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const = 0;
    virtual std::vector<std::unique_ptr<DocumentResiduals>> get_residuals(
            const uint64_t tenant,
            const std::vector<idx_t>& ids) const = 0;
    virtual std::vector<std::unique_ptr<DocumentMetadata>> get_metadata(
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