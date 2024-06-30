#ifndef FIELD_VALUE_H
#define FIELD_VALUE_H

#include <string>
#include <vector>
#include <variant>
#include <chrono>
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

namespace bitsery {


template<typename S>
void serialize(S& s, lintdb::FieldValue& fv) {
    s.value4b(fv.data_type);
    s.ext(fv.value, bitsery::ext::StdVariant{});
}

template<typename S>
void serialize(S& s, lintdb::DataType& dt) {
    s.value4b(dt);
}

template<typename S>
void serialize(S& s, lintdb::Tensor& tensor) {
    s.container(tensor, 1000); // Adjust size as needed
}

template<typename S>
void serialize(S& s, lintdb::TensorArray& tensorArray) {
    s.container(tensorArray, 100); // Adjust size as needed
}

template<typename S>
void serialize(S& s, lintdb::DateTime& dt) {
    s.ext(dt, bitsery::ext::StdChrono{});
}
}

namespace lintdb {

enum DataType {
    TENSOR,
    INTEGER,
    FLOAT,
    TEXT,
    DATETIME,
    _LEGACY_TEXT_MAP, /// This is a legacy type, and should not be used.
};

using Tensor = gsl::span<const float>;
using TensorArray = gsl::span<const float>;
using QuantizedTensor = std::vector<uint8_t>;
using QuantizedTensorArray = std::vector<QuantizedTensor>;
using DateTime = std::chrono::system_clock::time_point;

struct FieldValue {
    DataType data_type;
    std::variant<int, float, DateTime, Tensor, TensorArray, std::string> value;

    FieldValue(int v) : data_type(DataType::INTEGER), value(v) {}
    FieldValue(float v) : data_type(DataType::FLOAT), value(v) {}
    FieldValue(std::string v) : data_type(DataType::TEXT), value(v) {}
    FieldValue(DateTime v) : data_type(DataType::DATETIME), value(v) {}
    FieldValue(Tensor v) : data_type(DataType::TENSOR), value(v) {}
    FieldValue(TensorArray v) : data_type(DataType::TENSOR_ARRAY), value(v) {}
    FieldValue(std::map<std::string, std::string> v) : data_type(DataType::_LEGACY_TEXT_MAP), value(v) {}
};
}
#endif // FIELD_VALUE_H
