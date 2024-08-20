#ifndef FIELD_VALUE_H
#define FIELD_VALUE_H

#include <bitsery/adapter/stream.h>
#include <bitsery/bitsery.h>
#include <bitsery/ext/std_chrono.h>
#include <bitsery/ext/std_map.h>
#include <bitsery/ext/std_variant.h>
#include <bitsery/traits/string.h>
#include <bitsery/traits/vector.h>
#include <chrono>
#include <gsl/span>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
#include "lintdb/api.h"
#include "lintdb/assert.h"
#include "lintdb/api.h"
#include <json/reader.h>
#include <json/writer.h>

#define MAX_TENSOR_SIZE 10000
#define MAX_CENTROIDS_TO_STORE 40000000

namespace lintdb {

enum DataType {
    TENSOR,
    TENSOR_FLOAT16,
    QUANTIZED_TENSOR,
    INTEGER,
    FLOAT,
    FLOAT16,
    TEXT,
    DATETIME,
    COLBERT // colbert is our internal representation of colbert data. it
            // includes the residual codes and indexes.
};

inline std::string dataTypeToString(DataType type) {
    switch (type) {
        case TENSOR:
            return "TENSOR";
        case TENSOR_FLOAT16:
            return "TENSOR_FLOAT16";
        case QUANTIZED_TENSOR:
            return "QUANTIZED_TENSOR";
        case INTEGER:
            return "INTEGER";
        case FLOAT:
            return "FLOAT";
        case FLOAT16:
            return "FLOAT16";
        case TEXT:
            return "TEXT";
        case DATETIME:
            return "DATETIME";
        case COLBERT:
            return "COLBERT";
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

inline DataType stringToDataType(const std::string& str) {
    if (str == "TENSOR") {
        return TENSOR;
    } else if (str == "TENSOR_FLOAT16") {
        return TENSOR_FLOAT16;
    } else if (str == "QUANTIZED_TENSOR") {
        return QUANTIZED_TENSOR;
    } else if (str == "INTEGER") {
        return INTEGER;
    } else if (str == "FLOAT") {
        return FLOAT;
    } else if (str == "FLOAT16") {
        return FLOAT16;
    } else if (str == "TEXT") {
        return TEXT;
    } else if (str == "DATETIME") {
        return DATETIME;
    } else if (str == "COLBERT") {
        return COLBERT;
    } else {
        throw std::invalid_argument("Unknown DataType string: " + str);
    }
}

using Tensor = std::vector<float>;
using TensorFloat16 = std::vector<float16>;
using QuantizedTensor = std::vector<uint8_t>;
using Duration = std::chrono::duration<int64_t, std::milli>;
using DateTime = std::chrono::time_point<std::chrono::system_clock, Duration>;

struct ColBERTContextData {
    std::vector<idx_t> doc_codes;
    std::vector<uint8_t> doc_residuals;
};

/**
 * Because we're serializing SupportedTypes, the order of the variants matter.
 *
 * New types MUST be added at the bottom.
 */
using SupportedTypes = std::variant<
        idx_t,
        float,
        lintdb::DateTime,
        lintdb::Tensor,
        lintdb::QuantizedTensor,
        std::string,
        ColBERTContextData, // colbert is our internal representation of colbert
                           // data. it includes the residual codes and indexes.
        float16,
        lintdb::TensorFloat16
        >;

inline Json::Value supportedTypeToJSON(const SupportedTypes& st) {
    if(const auto st_float = std::get_if<float>(&st)) {
        return Json::Value(*st_float);
    } else if(const auto st_idx_t = std::get_if<idx_t>(&st)) {
        return Json::Value(*st_idx_t);
    } else if(const auto st_float16 = std::get_if<float16>(&st)) {
        return Json::Value(static_cast<float>(*st_float16));
    } else if(const auto st_string = std::get_if<std::string>(&st)) {
        return Json::Value(*st_string);
    } else if(const auto st_tensor = std::get_if<Tensor>(&st)) {
        Json::Value v;
        for (auto& t : *st_tensor) {
            v.append(t);
        }
        return v;
    } else if(const auto st_tensor_float16 = std::get_if<TensorFloat16>(&st)) {
        Json::Value v;
        for (auto& t : *st_tensor_float16) {
            v.append(static_cast<float>(t));
        }
        return v;
    } else if(const auto st_quantized_tensor = std::get_if<QuantizedTensor>(&st)) {
        Json::Value v;
        for(auto& t : *st_quantized_tensor) {
            v.append(t);
        }
        return v;
    } else if(const auto st_datetime = std::get_if<DateTime>(&st)) {
        auto time = st_datetime->time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
        return Json::Value(millis);
    } else if(const auto st_colbert = std::get_if<ColBERTContextData>(&st)) {
        Json::Value v;

        // Serialize doc_codes array
        Json::Value doc_codes(Json::arrayValue);
        for (auto& code : st_colbert->doc_codes) {
            doc_codes.append(code);
        }
        v["doc_codes"] = doc_codes;

        // Serialize doc_residuals array
        Json::Value doc_residuals(Json::arrayValue);
        for (auto& residual : st_colbert->doc_residuals) {
            doc_residuals.append(residual);
        }
        v["doc_residuals"] = doc_residuals;

        return v;
    }

    throw std::runtime_error("Unsupported type");
}


struct FieldValue {
    std::string name;
    lintdb::DataType data_type;
    size_t num_tensors = 0;
    SupportedTypes value;

    FieldValue() = default;

    FieldValue(std::string name, int v)
            : name(name), data_type(DataType::INTEGER), value(v) {}
    FieldValue(std::string name, float v)
            : name(name), data_type(DataType::FLOAT), value(v) {}
    FieldValue(std::string name, float16 v) : name(name), data_type(DataType::FLOAT16), value(v) {}
    FieldValue(std::string name, std::string v)
            : name(name), data_type(DataType::TEXT), value(v) {}
    FieldValue(std::string name, DateTime v)
            : name(name), data_type(DataType::DATETIME), value(v) {}
    FieldValue(std::string name, Tensor v)
            : name(name),
              data_type(DataType::TENSOR),
              num_tensors(1),
              value(v) {}
    FieldValue(std::string name, Tensor v, size_t num_tensors)
            : name(name),
              data_type(DataType::TENSOR),
              num_tensors(num_tensors),
              value(v) {}
    FieldValue(std::string name, TensorFloat16 v)
            : name(name),
              data_type(DataType::TENSOR_FLOAT16),
              num_tensors(1),
              value(v) {}
    FieldValue(std::string name, QuantizedTensor v, size_t num_tensors)
            : name(name),
              data_type(DataType::QUANTIZED_TENSOR),
              num_tensors(num_tensors),
              value(v) {}
    FieldValue(std::string name, ColBERTContextData v, size_t num_tensors)
            : name(name),
              data_type(DataType::COLBERT),
              num_tensors(num_tensors),
              value(v) {}

    Json::Value toJson() const {
        Json::Value root;
        root["name"] = name;
        root["data_type"] = dataTypeToString(data_type);
        root["num_tensors"] = static_cast<Json::UInt64>(num_tensors);

        // Serialize the SupportedTypes variant
        switch (data_type) {
            case DataType::INTEGER:
                root["value"] = std::get<int64_t>(value);
                break;
            case DataType::FLOAT:
                root["value"] = std::get<float>(value);
                break;
            case DataType::FLOAT16:
                root["value"] = std::get<float16>(value);
                break;
            case DataType::TEXT:
                root["value"] = std::get<std::string>(value);
                break;
            case DataType::DATETIME: {
                auto time = std::get<DateTime>(value).time_since_epoch();
                auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();

                root["value"] = millis;
                break;
            }
            case DataType::TENSOR: {
                Json::Value v;
                auto tensor = std::get<Tensor>(value);
                for( auto& t : tensor ) {
                    v.append(t);
                }
                root["value"] = v;
                break;
            }
            case DataType::TENSOR_FLOAT16: {
                Json::Value v;
                auto tensor = std::get<TensorFloat16>(value);
                for( auto& t : tensor ) {
                    v.append(static_cast<float>(t));
                }
                root["value"] = v;
                break;
            }
            case DataType::QUANTIZED_TENSOR: {
                Json::Value v;
                auto tensor = std::get<QuantizedTensor>(value);
                for( auto& t : tensor ) {
                    v.append(t);
                }
                root["value"] = v;
                break;
            }
            default:
                // Handle unknown data type if necessary
                break;
        }

        return root;
    }

    static FieldValue fromJson(const Json::Value &root) {
        FieldValue fieldValue;
        fieldValue.name = root["name"].asString();
        // convert a string data type to an enum
        fieldValue.data_type = stringToDataType(root["data_type"].asString());
        fieldValue.num_tensors = root["num_tensors"].asUInt64();

        // Deserialize the SupportedTypes variant
        switch (fieldValue.data_type) {
            case DataType::INTEGER:
                fieldValue.value = root["value"].asInt64();
                break;
            case DataType::FLOAT:
                fieldValue.value = root["value"].asFloat();
                break;
            case DataType::FLOAT16:
                fieldValue.value = static_cast<float16>(root["value"].asFloat());
                break;
            case DataType::TEXT:
                fieldValue.value = root["value"].asString();
                break;
            case DataType::DATETIME: {
                auto millis = root["value"].asInt64();
                auto time_point = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(
                        std::chrono::milliseconds(millis));
                fieldValue.value = DateTime(time_point);
                break;
            }
            case DataType::TENSOR: {
                Tensor tensor;
                for (const auto &v : root["value"]) {
                    tensor.push_back(v.asFloat());  // Assuming tensor is a vector of floats
                }
                fieldValue.value = tensor;
                break;
            }
            case DataType::TENSOR_FLOAT16: {
                TensorFloat16 tensor;
                for (const auto &v : root["value"]) {
                    tensor.push_back(static_cast<float16>(v.asFloat()));  // Assuming tensor is a vector of floats
                }
                fieldValue.value = tensor;
                break;
            }
            case DataType::QUANTIZED_TENSOR: {
                QuantizedTensor tensor;
                for (const auto &v : root["value"]) {
                    tensor.push_back(v.asInt());  // Assuming quantized tensor is a vector of ints
                }
                fieldValue.value = tensor;
                break;
            }
            default:
                // Handle unknown data type if necessary
                break;
        }

        return fieldValue;
    }
};

} // namespace lintdb

namespace bitsery {
template <typename S>
void serialize(S& s, lintdb::SupportedTypes& fv) {
    s.ext(fv,
          bitsery::ext::StdVariant{
                  [](S& p, float& o) { p.value4b(o); },
                  [](S& p, float16& o) {p.value2b(o);},
                  [](S& p, idx_t& o) { p.value8b(o); },
                  [](S& p, std::string& o) { p.text1b(o, 0xFFFF); },
                  [](S& p, lintdb::Tensor& o) {
                      p.container4b(o, MAX_TENSOR_SIZE);
                  },
                  [](S& p, lintdb::TensorFloat16 & o) {
                      p.container2b(o, MAX_TENSOR_SIZE);
                  },
                  [](S& p, lintdb::QuantizedTensor& o) {
                      p.container1b(o, MAX_TENSOR_SIZE);
                  },
                  [](S& p, lintdb::DateTime& o) {
                      p.ext8b(o, bitsery::ext::StdTimePoint{});
                  },
                  [](S& p, lintdb::ColBERTContextData& o) {
                      p.container8b(o.doc_codes, MAX_CENTROIDS_TO_STORE);
                      p.container1b(o.doc_residuals, MAX_CENTROIDS_TO_STORE);
                  }});
}

template <typename S>
void serialize(S& s, std::map<uint8_t, lintdb::SupportedTypes>& st) {
    s.ext(st,
          bitsery::ext::StdMap{256},
          [](S& s, uint8_t& key, lintdb::SupportedTypes& value) {
              s.value1b(key);
              s.ext(value,
                    bitsery::ext::StdVariant{
                            [](S& p, float& o) { p.value4b(o); },
                            [](S& p, float16& o) {p.value2b(o);},
                            [](S& p, idx_t& o) { p.value8b(o); },
                            [](S& p, std::string& o) { p.text1b(o, 0xFFFF); },
                            [](S& p, lintdb::Tensor& o) {
                                p.container4b(o, MAX_TENSOR_SIZE);
                            },
                            [](S& p, lintdb::TensorFloat16 & o) {
                                p.container2b(o, MAX_TENSOR_SIZE);
                            },
                            [](S& p, lintdb::QuantizedTensor& o) {
                                p.container1b(o, MAX_TENSOR_SIZE);
                            },
                            [](S& p, lintdb::DateTime& o) {
                                p.ext8b(o, bitsery::ext::StdTimePoint{});
                            },
                            [](S& p, lintdb::ColBERTContextData& o) {
                                p.container8b(
                                        o.doc_codes, MAX_CENTROIDS_TO_STORE);
                                p.container1b(
                                        o.doc_residuals,
                                        MAX_CENTROIDS_TO_STORE);
                            }});
          });
}

template <typename S>
void serialize(S& s, lintdb::QuantizedTensor& tensor) {
    s.container1b(tensor, MAX_TENSOR_SIZE);
}

template <typename S>
void serialize(S& s, std::vector<idx_t>& v) {
    s.container8b(v, MAX_CENTROIDS_TO_STORE);
}

template <typename S>
void serialize(S& s, lintdb::TensorFloat16 & v) {
    s.container2b(v, MAX_TENSOR_SIZE);
}

template <typename S>
void serialize(S& s, lintdb::Tensor& tensor) {
    s.container4b(tensor, MAX_TENSOR_SIZE); // Adjust size as needed
}

template <typename S>
void serialize(S& s, lintdb::DateTime& dt) {
    s.ext8b(dt, bitsery::ext::StdTimePoint{});
}

template <typename S>
void serialize(S& s, lintdb::ColBERTContextData data) {
    s.container8b(data.doc_codes, MAX_CENTROIDS_TO_STORE);
    s.container1b(data.doc_residuals, MAX_CENTROIDS_TO_STORE);
}
} // namespace bitsery

#endif // FIELD_VALUE_H
