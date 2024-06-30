#pragma once

/**
 * Schemas enable the index to intelligently index different types of data.
 * One of the main advantages of this is to
 */

#include <vector>
#include <string>
#include <map>
#include <stddef.h>
#include <json/json.h>


namespace lintdb {

/**
 * {
 *   fields: [
 *      {
 *          name: 'text',
 *          type: lintdb.TENSOR,
 *          params: {
 *              'dim': 128,
*          }
*       },
 *      {
 *          name: 'filter-x',
 *          type: 'text',
 *          params: {
 *              'analyzer': 'en',
 *          }
 */

enum FieldType {
    Indexed,
    Context,
    Stored,
    Analyzed
};

enum QuantizationType {
    NONE,
    BINARIZER,
    PQ,
};

enum LateInteractionType {
    COLBERT,
    XTR
};

struct FieldParameters {
    size_t dimensions;
    std::string analyzer;
    QuantizationType quantization;
    LateInteractionType lateInteractionType;
};

/**
 * A Document is made up of multiple fields.
 */
struct Field {
    std::string name; /// the name of the field
    DataType data_type; /// the data type. e.g. int, float, string, embedding.
    FieldType field_type; /// the field type. e.g. indexed or stored in the database.
    FieldParameters parameters; /// parameters for the field.

    Json::Value toJson() const;
    static Schema fromJson(const Json::Value& json);
};

/**
 * A schema dictates what data is stored, how it is stored, and the way we are
 * able to interact with the data.
 */
struct Schema {
    std::vector<Field> fields;

    Json::Value toJson() const;
    static Schema fromJson(const Json::Value& json);
};

} // namespace lintdb
