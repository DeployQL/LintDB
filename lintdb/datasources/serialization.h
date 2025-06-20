#pragma once

#include <map>
#include <string>
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/Schema.h"

namespace lintdb {

/**
 * @brief Serialize a single field value
 * @param field The field definition
 * @param value The value to serialize
 * @return Serialized string representation
 */
std::string serialize_field_value(const Field& field, const SupportedTypes& value);

/**
 * @brief Deserialize a single field value
 * @param field The field definition
 * @param data The serialized data
 * @return The deserialized value
 */
SupportedTypes deserialize_field_value(const Field& field, const std::string& data);

/**
 * @brief Serialize an entire document according to a schema
 * @param schema The schema defining the document structure
 * @param document Map of field names to values
 * @return Serialized string representation
 */
std::string serialize_document(
        const Schema& schema, 
        const std::map<std::string, SupportedTypes>& document);

/**
 * @brief Deserialize an entire document according to a schema
 * @param schema The schema defining the document structure
 * @param data The serialized data
 * @return Map of field names to values
 */
std::map<std::string, SupportedTypes> deserialize_document(
        const Schema& schema,
        const std::string& data);

} // namespace lintdb 