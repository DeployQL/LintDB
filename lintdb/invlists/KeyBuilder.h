#pragma once

#include <string>
#include <vector>
#include "lintdb/schema/DataTypes.h"
#include "lintdb/utils/endian.h"


namespace lintdb {
    /**
 *  * vector key: tenant::field::field_type::inverted_list::doc_id
 * text key: tenant::field::field_type::field_size::field_value::doc_id
 */
    class KeyBuilder {
    private:
        std::vector<unsigned char> data_;

    public:
        KeyBuilder& add(idx_t data) {
            store_bigendian(data, data_);

            return *this;
        }

        KeyBuilder& add(uint32_t data) {
            store_bigendian(data, data_);

            return *this;
        }

        KeyBuilder& add(uint64_t data) {
            store_bigendian(data, data_);

            return *this;
        }

        KeyBuilder& add(uint8_t data) {
            store_bigendian(data, data_);

            return *this;
        }

        KeyBuilder& add(float data) {
            uint32_t fbits = 0;
            memcpy(&fbits, &data, sizeof fbits);
            store_bigendian(fbits, data_);

            return *this;
        }

        KeyBuilder& add(DataType data) {
            store_bigendian<uint8_t>(data, data_);
            return *this;
        }

        KeyBuilder& add(DateTime data) {
            auto dv_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(data);
            auto epoch = dv_ms.time_since_epoch();

            int64_t epoch_ms = epoch.count();

            store_bigendian(epoch_ms, data_);
            return *this;
        }

        KeyBuilder& add(std::string data) {
            data_.insert(data_.end(), data.begin(), data.end());

            return *this;
        }

        std::string build() const {
            return {data_.begin(), data_.end()};
        }
    };

class StorageKey {
public:
    virtual idx_t doc_id() const = 0;

    virtual ~StorageKey() = default;
};

/**
 *  * vector key: tenant::field::field_type::inverted_list::doc_id
 * text key: tenant::field::field_type::field_size::field_value::doc_id
 */
class InvertedIndexKey: public StorageKey {
public:
    InvertedIndexKey(std::string& slice) {
        auto ptr = slice.data();
        tenant_ = load_bigendian<uint64_t>(ptr);
        field_ = load_bigendian<uint8_t>(ptr+sizeof(tenant_));
        auto field_type = load_bigendian<uint8_t>(ptr+sizeof(tenant_)+sizeof(field_));
        auto type = DataType(field_type);

        idx_t doc;
        switch(type) {
            // these types expect idx_t in the key.
            case DataType::INTEGER:
            case DataType::DATETIME:
            case DataType::QUANTIZED_TENSOR:
            case DataType::TENSOR:
            case DataType::FLOAT: {
                auto invlist = load_bigendian<idx_t>(ptr+sizeof(tenant_) + sizeof(field_) + sizeof(field_type));
                doc_id_ = load_bigendian<idx_t>(ptr+sizeof(tenant_) + sizeof(field_) + sizeof(field_type) + sizeof(invlist));
            }
            // Text keys encode the length and the string into the key.
            case DataType::TEXT: {
                auto ptr_to_size = ptr+sizeof(tenant_) + sizeof(field_) + sizeof(field_type);
                auto field_size = load_bigendian<uint32_t>(ptr_to_size);
                doc_id_ = load_bigendian<idx_t>(ptr_to_size + sizeof(field_size) + field_size);
            }
        }

    }

    uint64_t tenant() const {
        return tenant_;
    }

    uint8_t field() const {
        return field_;
    }

    idx_t doc_id() const override {
        return doc_id_;
    }

private:
    uint64_t tenant_;
    uint8_t field_;
    idx_t doc_id_;
};

/**
 * For any field type, we'll have the following:
 * tenant::field::doc_id
 */
class ContextKey: public StorageKey {
public:
    ContextKey(std::string& slice) {
        auto ptr = slice.data();

        tenant_ = load_bigendian<uint64_t>(ptr);
        field_ = load_bigendian<uint8_t>(ptr+sizeof(tenant_));
        doc_id_ = load_bigendian<idx_t>(ptr+sizeof(tenant_)+sizeof(field_));
    }

    idx_t doc_id() const override {
        return doc_id_;
    }
    uint64_t tenant() const {
        return tenant_;
    }

    uint8_t field() const {
        return field_;
    }

    idx_t doc_id() const override {
        return doc_id_;
    }

private:
    uint64_t tenant_;
    uint8_t field_;
    idx_t doc_id_;
};

/**
 * ForwardIndexKey looks up values by docid.
 *
 * tenant::doc_id.
 */
class ForwardIndexKey: public StorageKey {
public:
    ForwardIndexKey(std::string& slice) {
        auto ptr = slice.data();
        tenant_ = load_bigendian<uint64_t>(ptr);
        doc_id_ = load_bigendian<idx_t>(ptr + sizeof(tenant_));
    }

    uint64_t tenant() const {
        return tenant_;
    }

    idx_t doc_id() const override {
        return doc_id_;
    }

private:
    uint64_t tenant_;
    idx_t doc_id_;
};
}
