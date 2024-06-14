#ifndef LINTDB_INVLISTS_UTIL_H
#define LINTDB_INVLISTS_UTIL_H

#include <rocksdb/db.h>
#include <string>
#include <vector>
#include "lintdb/api.h"

using namespace std;

namespace lintdb {
/**
 * DEPRECATED: Please use TokenKey instead.
 *
 * A Key is composed of several integers that specify a specific document in the
 * inverse index.
 *
 */
struct Key {
    uint64_t tenant;
    idx_t inverted_list_id;
    idx_t id;
    bool exclude_id;

    std::string serialize() const;

    static Key from_slice(const rocksdb::Slice& slice);
};

/**
 * TokenKey is an inverted list key that maps to the token of a document.
 *
 * The total key is: tenant::inverted_list_id::doc_id::doc_token_id.
 *
 * When searching against this key, exclude_id removes both the doc_id and doc_token_id.
 */
struct TokenKey {
    uint64_t tenant;
    idx_t inverted_list_id;
    idx_t doc_id;
    idx_t token_id;
    bool exclude_id;

    std::string serialize() const;

    static TokenKey from_slice(const rocksdb::Slice& slice);
};

/**
 * ForwardIndexKey is a struct that represents a key in the forward index.
 */
struct ForwardIndexKey {
    uint64_t tenant;
    idx_t id;

    std::string serialize() const;

    static ForwardIndexKey from_slice(const rocksdb::Slice& slice);
};

template <typename T>
T load_bigendian(void const* bytes) {
    T num = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        num |= static_cast<T>(static_cast<const unsigned char*>(bytes)[i])
                << (8 * (sizeof(T) - i - 1));
    }
    return num;
}

template <typename T>
void store_bigendian(T num, std::vector<unsigned char>& bigEndian) {
    for (int i = sizeof(T) - 1; i >= 0; i--) {
        bigEndian.push_back((num >> (8 * i)) & 0xff);
    }
}
} // namespace lintdb

#endif
