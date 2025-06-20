#include "serialization.h"
#include <bitsery/adapter/buffer.h>
#include <bitsery/bitsery.h>
#include <glog/logging.h>
#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/Schema.h"

namespace lintdb {

using Buffer = std::vector<uint8_t>;
using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;
using InputAdapter = bitsery::InputBufferAdapter<std::string>;

std::string serialize_field_value(const Field& field, const SupportedTypes& value) {
    Buffer buf;
    auto written = bitsery::quickSerialization(OutputAdapter{buf}, value);
    return std::string(buf.begin(), buf.begin() + written);
}

SupportedTypes deserialize_field_value(const Field& field, const std::string& data) {
    SupportedTypes res;
    auto state = bitsery::quickDeserialization<InputAdapter>(
            {data.begin(), data.size()}, res);
    if (state.first != bitsery::ReaderError::NoError || !state.second) {
        throw std::runtime_error("Failed to deserialize field value");
    }
    return res;
}

std::string serialize_document(const Schema& schema, const std::map<std::string, SupportedTypes>& document) {
    // Create a map of field_id to value for serialization
    std::map<uint8_t, SupportedTypes> field_values;
    
    for (const auto& [field_name, value] : document) {
        try {
            const Field& field = schema.get_field(field_name);
            field_values[field.field_id] = value;
        } catch (const std::runtime_error& e) {
            LOG(WARNING) << "Field " << field_name << " not found in schema, skipping";
        }
    }

    Buffer buf;
    auto written = bitsery::quickSerialization(OutputAdapter{buf}, field_values);
    return std::string(buf.begin(), buf.begin() + written);
}

std::map<std::string, SupportedTypes> deserialize_document(
        const Schema& schema, 
        const std::string& data) {
    std::map<uint8_t, SupportedTypes> field_values;
    auto state = bitsery::quickDeserialization<InputAdapter>(
            {data.begin(), data.size()}, field_values);
    
    if (state.first != bitsery::ReaderError::NoError || !state.second) {
        throw std::runtime_error("Failed to deserialize document");
    }

    // Convert field_id map back to field_name map
    std::map<std::string, SupportedTypes> result;
    for (const auto& [field_id, value] : field_values) {
        // Find the field name for this field_id
        for (const auto& field : schema.fields) {
            if (field.field_id == field_id) {
                result[field.name] = value;
                break;
            }
        }
    }

    return result;
}

} // namespace lintdb 