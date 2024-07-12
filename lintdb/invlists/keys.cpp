
#include "lintdb/invlists/keys.h"
#include <glog/logging.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "lintdb/assert.h"
#include "lintdb/utils/endian.h"

namespace lintdb {
std::string Key::serialize() const {
    std::vector<unsigned char> serialized_str;

    store_bigendian(this->tenant, serialized_str);
    store_bigendian(this->inverted_list_id, serialized_str);

    if (!exclude_id) {
        store_bigendian(this->id, serialized_str);
    }

    std::string result =
            std::string(serialized_str.begin(), serialized_str.end());

    return result;
}

Key Key::from_slice(const rocksdb::Slice& slice) {
    // tenant is 8 bytes, inverted_list_id is 8 bytes, and id is 8 bytes.
    // slices must have an id assigned.
    LINTDB_THROW_IF_NOT(slice.size() > 16);

    auto key_ptr = slice.data();
    uint64_t tenant = load_bigendian<uint64_t>(key_ptr);
    idx_t inverted_list_id = load_bigendian<idx_t>(key_ptr + sizeof(tenant));
    idx_t id = load_bigendian<idx_t>(
            key_ptr + sizeof(tenant) + sizeof(inverted_list_id));

    return Key{tenant, inverted_list_id, id};
}

std::string ForwardIndexKey::serialize() const {
    // size_t key_size = sizeof(this->tenant) + sizeof(this->id);
    std::vector<unsigned char> serialized_str;

    store_bigendian(this->tenant, serialized_str);
    store_bigendian(this->id, serialized_str);

    return std::string(serialized_str.begin(), serialized_str.end());
}

ForwardIndexKey ForwardIndexKey::from_slice(const rocksdb::Slice& slice) {
    // 8 bytes are used for the prefix. tenant is 8 bytes, and id is 8 bytes.
    // slices must have an id assigned.
    LINTDB_THROW_IF_NOT(slice.size() > 8);

    auto key_ptr = slice.data();
    uint64_t tenant = load_bigendian<uint64_t>(key_ptr);
    idx_t id = load_bigendian<idx_t>(key_ptr + sizeof(tenant));

    return ForwardIndexKey{tenant, id};
}

std::string TokenKey::serialize() const {
    std::vector<unsigned char> serialized_str;

    store_bigendian<uint64_t>(this->tenant, serialized_str);
    store_bigendian<uint8_t>(this->field, serialized_str);
    store_bigendian<idx_t>(this->inverted_list_id, serialized_str);

    if (!exclude_id) {
        store_bigendian<idx_t>(this->doc_id, serialized_str);
        store_bigendian<idx_t>(this->token_id, serialized_str);
    }

    std::string result =
            std::string(serialized_str.begin(), serialized_str.end());

    return result;
}
TokenKey TokenKey::from_slice(const rocksdb::Slice& slice) {
    // tenant is 8 bytes, inverted_list_id is 8 bytes, and id is 8 bytes.
    // slices must have an id assigned.
    LINTDB_THROW_IF_NOT(slice.size() > 16);

    auto key_ptr = slice.data();
    auto tenant = load_bigendian<uint64_t>(key_ptr);
    auto field = load_bigendian<uint8_t>(key_ptr + sizeof(tenant));
    auto inverted_list_id = load_bigendian<idx_t>(key_ptr + sizeof(tenant) + sizeof(field));
    auto doc_id = load_bigendian<idx_t>(
            key_ptr + sizeof(tenant) + sizeof(inverted_list_id)+ sizeof(field));

    auto token_id = load_bigendian<idx_t>(
            key_ptr + sizeof(tenant) + sizeof(field) + sizeof(inverted_list_id) +
            sizeof(doc_id));

    return TokenKey{tenant, field, inverted_list_id, doc_id, token_id};
}

std::string ContextKey::serialize() const {
    // size_t key_size = sizeof(this->tenant) + sizeof(this->id);
    std::vector<unsigned char> serialized_str;

    store_bigendian(this->tenant, serialized_str);
    store_bigendian(this->field, serialized_str);
    if(exclude_id) {
        store_bigendian(this->id, serialized_str);
    }

    return std::string(serialized_str.begin(), serialized_str.end());
}

ContextKey ContextKey::from_slice(const rocksdb::Slice& slice) {
    // 8 bytes are used for the prefix. tenant is 8 bytes, and id is 8 bytes.
    // slices must have an id assigned.
    LINTDB_THROW_IF_NOT(slice.size() > 8);

    auto key_ptr = slice.data();
    uint64_t tenant = load_bigendian<uint64_t>(key_ptr);
    uint8_t field = load_bigendian<uint8_t>(key_ptr + sizeof(tenant));
    idx_t id = load_bigendian<idx_t>(key_ptr + sizeof(tenant) + sizeof(field));

    return ContextKey{tenant, field, id};
}
} // namespace lintdb