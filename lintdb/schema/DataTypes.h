#ifndef FIELD_VALUE_H
#define FIELD_VALUE_H

#include <string>
#include <vector>
#include <variant>
#include <chrono>
#include <map>
#include <stdexcept>
#include <bitsery/bitsery.h>
#include <bitsery/ext/std_variant.h>
#include <bitsery/ext/std_chrono.h>
#include <bitsery/bitsery.h>
#include <bitsery/adapter/stream.h>
#include <bitsery/traits/string.h>
#include <bitsery/traits/vector.h>
#include <bitsery/ext/std_variant.h>
#include <bitsery/ext/std_chrono.h>
#include <bitsery/ext/std_map.h>
#include <gsl/span>
#include "lintdb/api.h"
#include "lintdb/assert.h"


#define MAX_TENSOR_SIZE 10000
#define MAX_CENTROIDS_TO_STORE 40000000

namespace lintdb {

enum DataType {
    TENSOR,
    QUANTIZED_TENSOR,
    INTEGER,
    FLOAT,
    TEXT,
    DATETIME,
    COLBERT // colbert is our internal representation of colbert data. it includes the residual codes and indexes.
};

const std::unordered_map<int, DataType> IntToDataType = {
        {0, DataType::TENSOR},
        {1, DataType::QUANTIZED_TENSOR},
        {2, DataType::INTEGER},
        {3, DataType::FLOAT},
        {4, DataType::TEXT},
        {5, DataType::DATETIME}
};

const std::unordered_map<DataType, int> DataTypeToInt = {
        {DataType::TENSOR, 0},
        {DataType::QUANTIZED_TENSOR, 1},
        {DataType::INTEGER, 2},
        {DataType::FLOAT, 3},
        {DataType::TEXT, 4},
        {DataType::DATETIME, 5}
};

using Tensor = std::vector< float>;
using QuantizedTensor = std::vector<uint8_t>;
using Duration = std::chrono::duration<int64_t, std::milli>;
using DateTime = std::chrono::time_point<std::chrono::system_clock, Duration>;

struct ColBERTContextData {
    std::vector<idx_t> doc_codes;
    std::vector<uint8_t>  doc_residuals;
};

using SupportedTypes = std::variant<idx_t, float, lintdb::DateTime, lintdb::Tensor, lintdb::QuantizedTensor, std::string, ColBERTContextData>;

struct FieldValue {
    std::string name;
    lintdb::DataType data_type;
    size_t num_tensors = 0;
    SupportedTypes value;

    FieldValue() = default;

    FieldValue(std::string name, int v) : name(name), data_type(DataType::INTEGER), value(v) {}
    FieldValue(std::string name, float v) : name(name),  data_type(DataType::FLOAT), value(v) {}
    FieldValue(std::string name, std::string v) : name(name),  data_type(DataType::TEXT), value(v) {}
    FieldValue(std::string name, DateTime v) : name(name), data_type(DataType::DATETIME), value(v) {}
    FieldValue(std::string name, Tensor v) : name(name), data_type(DataType::TENSOR), num_tensors(1), value(v) {}
    FieldValue(std::string name, Tensor v, size_t num_tensors) : name(name), data_type(DataType::TENSOR), num_tensors(num_tensors), value(v) {}
    FieldValue(std::string name, QuantizedTensor v, size_t num_tensors) : name(name), data_type(DataType::QUANTIZED_TENSOR), num_tensors(num_tensors), value(v) {}
    FieldValue(std::string name, ColBERTContextData v, size_t num_tensors) : name(name), data_type(DataType::COLBERT), num_tensors(num_tensors), value(v) {}
};

}

namespace bitsery {
    template<typename S>
    void serialize(S& s, lintdb::SupportedTypes& fv) {
        s.ext(fv, bitsery::ext::StdVariant{
                [](S& p, float& o) {
                    p.value4b(o);
                },
                [](S& p, idx_t& o) {
                    p.value8b(o);
                },
                [](S& p, std::string& o) {
                    p.text1b(o, 0xFFFF);
                },
                [](S& p, lintdb::Tensor& o) {
                    p.container4b(o, MAX_TENSOR_SIZE);
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
                }
        });
    }

    template<typename S>
    void serialize(S& s, std::map<uint8_t, lintdb::SupportedTypes>& st) {
        s.ext(st,
              bitsery::ext::StdMap{256},
              [](S& s, uint8_t& key, lintdb::SupportedTypes& value) {
                  s.value1b(key);
                  s.ext(value, bitsery::ext::StdVariant{
                          [](S& p, float& o) {
                              p.value4b(o);
                          },
                          [](S& p, idx_t& o) {
                              p.value8b(o);
                          },
                          [](S& p, std::string& o) {
                              p.text1b(o, 0xFFFF);
                          },
                          [](S& p, lintdb::Tensor& o) {
                              p.container4b(o, MAX_TENSOR_SIZE);
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
                            }
                  });
              });
    }

    template<typename S>
    void serialize(S& s, lintdb::QuantizedTensor& tensor) {
        s.container1b(tensor, MAX_TENSOR_SIZE);
    }

    template<typename S>
    void serialize(S& s, std::vector<idx_t>& v) {
        s.container8b(v, MAX_CENTROIDS_TO_STORE);
    }

    template<typename S>
    void serialize(S& s, lintdb::Tensor& tensor) {
        s.container4b(tensor, MAX_TENSOR_SIZE); // Adjust size as needed
    }

    template<typename S>
    void serialize(S& s, lintdb::DateTime& dt) {
        s.ext8b(dt, bitsery::ext::StdTimePoint{});
    }

    template<typename S>
    void serialize(S& s, lintdb::ColBERTContextData data) {
        s.container8b(data.doc_codes, MAX_CENTROIDS_TO_STORE);
        s.container1b(data.doc_residuals, MAX_CENTROIDS_TO_STORE);
    }
}

#endif // FIELD_VALUE_H
