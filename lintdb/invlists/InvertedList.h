#ifndef LINTDB_INVLISTS_INVERTED_LIST_H
#define LINTDB_INVLISTS_INVERTED_LIST_H

#include <stddef.h>
#include <memory>
#include <vector>
#include "lintdb/api.h"
#include "lintdb/constants.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/Iterator.h"

namespace lintdb {
/**
 * InvertedList manages the storage of centroid -> codes mappping.
 *
 * It's also handling forward index behavior, and we should eventually split
 * this out into a separate class.
 */
struct InvertedList {
    virtual void add(const uint64_t tenant, EncodedDocument* doc) = 0;
    virtual void remove(const uint64_t tenant, std::vector<idx_t> ids) = 0;
    virtual void merge(
            rocksdb::DB* db,
            std::vector<rocksdb::ColumnFamilyHandle*> cfs) = 0;

    virtual std::unique_ptr<Iterator> get_iterator(
            const uint64_t tenant,
            const idx_t inverted_list) const = 0;

    virtual std::vector<idx_t> get_mapping(const uint64_t tenant, idx_t id)
            const = 0;

    virtual ~InvertedList() = default;
};

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

    virtual void add(
            const uint64_t tenant,
            EncodedDocument* doc,
            bool store_codes) = 0;
    virtual void remove(const uint64_t tenant, std::vector<idx_t> ids) = 0;

    virtual void merge(
            rocksdb::DB* db,
            std::vector<rocksdb::ColumnFamilyHandle*> cfs) = 0;

    virtual ~ForwardIndex() = default;
};
} // namespace lintdb

#endif