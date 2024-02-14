
#include "lintdb/invlists/util.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include "lintdb/assert.h"

namespace lintdb {
   std::string Key::serialize() const {
        size_t key_size = sizeof(this->tenant) + sizeof(this->inverted_list_id) + (exclude_id ? 0 : sizeof(this->id));
        std::vector<unsigned char> serialized_str;

        store_bigendian(this->tenant, serialized_str);
        store_bigendian(this->inverted_list_id, serialized_str);

        if (!exclude_id) {
            store_bigendian(this->id, serialized_str);
        }

       return std::string(serialized_str.begin(), serialized_str.end());
    }

    Key Key::from_slice(const rocksdb::Slice &slice) {
        // 12 bytes are used for the prefix. tenant is 8 bytes, inverted_list_id is 4 bytes, and id is 8 bytes.
        // slices must have an id assigned.
        LINTDB_THROW_IF_NOT(slice.size() > 12);

        auto key_ptr = slice.data();
        uint64_t tenant = load_bigendian<uint64_t>(key_ptr);
        size_t inverted_list_id = load_bigendian<size_t>(key_ptr + sizeof(tenant));
        idx_t id = load_bigendian<idx_t>(key_ptr + sizeof(tenant) + sizeof(inverted_list_id));

        return Key {
            tenant,
            inverted_list_id,
            id
        };
    }

    std::string ForwardIndexKey::serialize() const {
        size_t key_size = sizeof(this->tenant) + sizeof(this->id);
        std::vector<unsigned char> serialized_str;

        store_bigendian(this->tenant, serialized_str);
        store_bigendian(this->id, serialized_str);

       return std::string(serialized_str.begin(), serialized_str.end());
    }

    ForwardIndexKey ForwardIndexKey::from_slice(const rocksdb::Slice &slice) {
        // 8 bytes are used for the prefix. tenant is 8 bytes, and id is 8 bytes.
        // slices must have an id assigned.
        LINTDB_THROW_IF_NOT(slice.size() > 8);

        auto key_ptr = slice.data();
        uint64_t tenant = load_bigendian<uint64_t>(key_ptr);
        idx_t id = load_bigendian<idx_t>(key_ptr + sizeof(tenant));

        return ForwardIndexKey {
            tenant,
            id
        };
    }

   
}