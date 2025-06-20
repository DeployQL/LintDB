#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lintdb/schema/Schema.h"

namespace lintdb {

/**
 * KeyValueIterator provides an interface for iterating over key-value pairs.
 */
class KeyValueIterator {
public:
    virtual ~KeyValueIterator() = default;

    /**
     * Returns true if the iterator is valid.
     */
    virtual bool is_valid() const = 0;

    /**
     * Advances the iterator to the next key-value pair.
     */
    virtual void advance() = 0;

    /**
     * Returns the current key.
     */
    virtual std::string key() const = 0;

    /**
     * Returns the current value as a string.
     */
    virtual std::string value() const = 0;
};

/**
 * DataSource provides a read-only interface for accessing data.
 * It abstracts the underlying storage mechanism (e.g., RocksDB) and provides
 * a key-value interface for data access.
 */
class DataSource {
public:
    virtual ~DataSource() = default;

    /**
     * Get the schema of the data source.
     */
    virtual Schema schema() const = 0;

    /**
     * Get an iterator over all key-value pairs.
     */
    virtual std::unique_ptr<KeyValueIterator> scan(uint64_t tenant_id) = 0;

    /**
     * Get an iterator over key-value pairs with a prefix.
     */
    virtual std::unique_ptr<KeyValueIterator> scan_prefix(const std::string& prefix) = 0;

    /**
     * Get a value by key.
     */
    virtual std::string get(const std::string& key) = 0;

    /**
     * Get an iterator over key-value pairs for a specific tenant and column.
     */
    virtual std::unique_ptr<KeyValueIterator> get_iterator(uint64_t tenant_id, const std::string& column_name) {
        // Default implementation uses scan and filters by prefix
        std::string prefix = std::to_string(tenant_id) + ":" + column_name + ":";
        return scan_prefix(prefix);
    }
}; 
}