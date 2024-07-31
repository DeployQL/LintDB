#pragma once

#include <vector>
#include <string>
#include <map>
#include <stddef.h>
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/schema/DataTypes.h"
#include <json/json.h>


namespace lintdb {


enum class FieldType {
    Indexed,
    Context,
    Stored,
    Colbert
};

struct FieldParameters {
    size_t dimensions = 0;
    std::string analyzer = "";
    QuantizerType quantization = QuantizerType::UNKNOWN;
    size_t num_centroids = 0;
    size_t num_iterations = 10;
    size_t num_subquantizers = 0; // used for PQ quantizer
    size_t nbits = 1; // used for PQ quantizer
};

/**
 * A Schema is made up of multiple fields.
 */
struct Field {
    std::string name; /// the name of the field
    DataType data_type; /// the data type. e.g. int, float, string, embedding.
    std::vector<FieldType> field_types; /// the field types. e.g. indexed or stored in the database.
    FieldParameters parameters; /// parameters for the field.

    Field() = default;
    Field(const std::string& name, const DataType data_type, const std::vector<FieldType>& field_types, const FieldParameters& parameters)
        : name(name), data_type(data_type), field_types(field_types), parameters(parameters) {}

    Json::Value toJson() const;
    static Field fromJson(const Json::Value& json);

    void add_field_type(FieldType field_type) {
        field_types.push_back(field_type);
    }
};

struct IndexedField : public Field {
    IndexedField(const std::string& name, const DataType data_type, const FieldParameters& parameters)
        : Field(name, data_type, {FieldType::Indexed}, parameters) {}
};

struct ContextField : public Field {
    ContextField(const std::string& name, const DataType data_type, const FieldParameters& parameters)
        : Field(name, data_type, {FieldType::Context}, parameters) {}
};

struct StoredField : public Field {
    StoredField(const std::string& name, const DataType data_type, const FieldParameters& parameters)
        : Field(name, data_type, {FieldType::Stored}, parameters) {}
};

struct ColbertField : public Field {
    ColbertField(const std::string& name, const DataType data_type, const FieldParameters& parameters)
        : Field(name, data_type, {FieldType::Colbert}, parameters) {}
};

/**
 * A schema dictates what data is stored, how it is stored, and the way we are
 * able to interact with the data.
 */
struct Schema {
    std::vector<Field> fields;

    Schema() = default;
    explicit Schema(const std::vector<Field>& fields) : fields(fields) {}

    Json::Value toJson() const;
    static Schema fromJson(const Json::Value& json);

    inline void add_field(Field& field) {
        fields.push_back(field);
    }
};

} // namespace lintdb
