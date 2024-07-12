#pragma once

#include <vector>
#include <string>
#include <map>
#include <stddef.h>
#include <json/json.h>
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/schema/DataTypes.h"

namespace lintdb {

enum FieldType {
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

    Json::Value toJson() const;
    static Field fromJson(const Json::Value& json);
};

/**
 * A schema dictates what data is stored, how it is stored, and the way we are
 * able to interact with the data.
 */
struct Schema {
    std::vector<Field> fields;

    Json::Value toJson() const;
    static Schema fromJson(const Json::Value& json);

    inline void add_field(Field& field) {
        fields.push_back(field);
    }
};

} // namespace lintdb
