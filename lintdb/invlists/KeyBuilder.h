#pragma once

#include <string>
#include <cstring>
#include <vector>
#include "lintdb/schema/DataTypes.h"
#include "lintdb/utils/endian.h"
#include <glog/logging.h>
#include <chrono>

namespace lintdb {
    /**
 *  * vector key: tenant::field::field_type::inverted_list::doc_id
 * text key: tenant::field::field_type::field_size::field_value::doc_id
 */
    class KeyBuilder {
    private:
        std::vector<unsigned char> data_;

    public:
        KeyBuilder &add(DateTime data) {
            auto dv_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(data);
            auto epoch = dv_ms.time_since_epoch();

            int64_t epoch_ms = epoch.count();

            store_bigendian(epoch_ms, data_);
            return *this;
        }

        KeyBuilder &add(DataType data) {
            store_bigendian<uint8_t>(data, data_);
            return *this;
        }

        template<typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
        KeyBuilder& add(T data) {
            store_bigendian(data, data_);
            return *this;
        }

        KeyBuilder &add(float data) {
            uint32_t fbits = 0;
            memcpy(&fbits, &data, sizeof fbits);
            store_bigendian(fbits, data_);

            return *this;
        }

        KeyBuilder &add(std::string data) {
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
 * InvertedIndexKey represents a deserialized key from the inverted index.
 *
 * It can get a little complicated, because at runtime, we won't know what type of field value we're
 * looking at, but a field value could any supported type.
 *
 *  * vector key: tenant::field::field_type::inverted_list::doc_id
 * text key: tenant::field::field_type::field_size::field_value::doc_id
 */
    class InvertedIndexKey : public StorageKey {
    public:
        InvertedIndexKey() = default;
        InvertedIndexKey(uint64_t tenant, uint8_t field, DataType type, SupportedTypes& value, idx_t doc_id)
                : tenant_(tenant), field_(field), doc_id_(doc_id), field_value_(value) {
        }

        explicit InvertedIndexKey(std::string &slice) {
            auto ptr = slice.data();
            tenant_ = load_bigendian<uint64_t>(ptr);
            field_ = load_bigendian<uint8_t>(ptr + sizeof(tenant_));
            auto field_type = load_bigendian<uint8_t>(ptr + sizeof(tenant_) + sizeof(field_));
            auto type = DataType(field_type);

            switch (type) {
                // these types expect an 8 byte value.
                case DataType::DATETIME: {
                    auto val = load_bigendian<uint64_t>(ptr + sizeof(tenant_) + sizeof(field_) + sizeof(field_type));
                    auto dur = DateTime(Duration(val));

                    field_value_ = dur;
                    doc_id_ = load_bigendian<idx_t>(
                            ptr + sizeof(tenant_) + sizeof(field_) + sizeof(field_type) + sizeof(val));
                    break;
                }
                case DataType::INTEGER:
                case DataType::QUANTIZED_TENSOR: /// stores the inverted list idx of the tensor.
                case DataType::TENSOR: /// stores the inverted list idx of the tensor.
                case DataType::FLOAT: {
                    // note that we aren't taking size of field_value_ since we're using std::variant.
                    field_value_ = load_bigendian<idx_t>(ptr + sizeof(tenant_) + sizeof(field_) + sizeof(field_type));
                    doc_id_ = load_bigendian<idx_t>(
                            ptr + sizeof(tenant_) + sizeof(field_) + sizeof(field_type) + sizeof(idx_t));
                    break;
                }
                    // Text keys encode the length and the string into the key.
                case DataType::TEXT: {
                    auto field_size = load_bigendian<uint32_t>(ptr + sizeof(tenant_) + sizeof(field_) + sizeof(field_type));
                    field_value_ = std::string(ptr + sizeof(tenant_) + sizeof(field_) + sizeof(field_type) + sizeof(field_size), field_size);
                    doc_id_ = load_bigendian<idx_t>(ptr + sizeof(tenant_) + sizeof(field_) + sizeof(field_type) + sizeof(field_size) + field_size);
                    break;
                }
                default:
                    throw LintDBException("type not regonized in inverted index");
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

        SupportedTypes field_value() const {
            return field_value_;
        }

    private:
        uint64_t tenant_;
        uint8_t field_;
        idx_t doc_id_;
        SupportedTypes field_value_;
    };

/**
 * For any field type, we'll have the following:
 * tenant::field::doc_id
 */
    class ContextKey : public StorageKey {
    public:
        ContextKey() = default;

        explicit ContextKey(std::string &slice) {
            auto ptr = slice.data();

            tenant_ = load_bigendian<uint64_t>(ptr);
            field_ = load_bigendian<uint8_t>(ptr + sizeof(tenant_));
            doc_id_ = load_bigendian<idx_t>(ptr + sizeof(tenant_) + sizeof(field_));
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
    class ForwardIndexKey : public StorageKey {
    public:
        ForwardIndexKey() = default;

        explicit ForwardIndexKey(std::string &slice) {
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

    inline std::string create_index_id(uint64_t tenant, uint8_t field, DataType type, SupportedTypes value, idx_t doc_id) {
        KeyBuilder kb;
        kb.add(tenant).add(field).add(type);

        switch(type) {
            case DataType::INTEGER: {
                kb.add(std::get<idx_t>(value));
                break;
            }
            case DataType::DATETIME: {
                DateTime  dt = std::get<DateTime>(value);
                kb.add(dt);
                break;
            }
            case DataType::FLOAT:
                kb.add(std::get<float>(value));
                break;
            case DataType::QUANTIZED_TENSOR:
            case DataType::TENSOR:
                // tensors store the inverted index list, so we should expect an idx_t.
                kb.add(std::get<idx_t>(value));
                break;
            case DataType::TEXT: {
                std::string v = std::get<std::string>(value);
                kb.add(uint32_t(v.size())).add(v);
                break;
            }
            default:
                throw LintDBException("type not recognized");
        }

        kb.add(doc_id);

        return kb.build();
    }

    inline std::string create_index_prefix(uint64_t tenant, uint8_t field, DataType type, SupportedTypes value) {
        KeyBuilder kb;
        kb.add(tenant).add(field).add(type);

        switch(type) {
            case DataType::INTEGER: {
                kb.add(std::get<idx_t>(value));
                break;
            }
            case DataType::DATETIME: {
                kb.add(std::get<DateTime>(value));
                break;
            }
            case DataType::FLOAT:
                kb.add(std::get<float>(value));
                break;
            case DataType::QUANTIZED_TENSOR:
            case DataType::TENSOR:
                // tensors store the inverted index list, so we should expect an idx_t.
                kb.add(std::get<idx_t>(value));
                break;
            case DataType::TEXT: {
                std::string v = std::get<std::string>(value);
                kb.add(uint32_t(v.size())).add(v);
                break;
            }
            default:
                throw LintDBException("type not recognized");
        }

        return kb.build();
    }

    inline std::string create_forward_index_id(uint64_t tenant, idx_t doc_id) {
        KeyBuilder kb;
        kb.add(tenant).add(doc_id);

        return kb.build();
    }

    inline std::string create_context_prefix(uint64_t tenant, uint8_t field) {
        KeyBuilder kb;
        kb.add(tenant).add(field);

        return kb.build();
    }

    inline std::string create_context_id(uint64_t tenant, uint8_t field, idx_t doc_id) {
        KeyBuilder kb;
        kb.add(tenant).add(field).add(doc_id);

        return kb.build();
    }
}



