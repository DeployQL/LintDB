#include "Schema.h"


namespace lintdb {
Json::Value Field::toJson() const {
    Json::Value json;
    json["name"] = name;
    json["data_type"] = static_cast<int>(data_type);
    json["field_type"] = static_cast<int>(field_type);

    Json::Value params;
    params["dimensions"] = parameters.dimensions;
    params["analyzer"] = parameters.analyzer;
    params["quantization_type"] = static_cast<int>(parameters.quantization_type);
    json["parameters"] = params;

    return json;
}

Field Field::fromJson(const Json::Value& json) {
    Field field;
    field.name = json["name"].asString();
    field.data_type = static_cast<DataType>(json["data_type"].asInt());
    field.field_type = static_cast<FieldType>(json["field_type"].asInt());

    const Json::Value& params = json["parameters"];
    field.parameters.dimensions = params["dimensions"].asUInt();
    field.parameters.analyzer = params["analyzer"].asString();
    field.parameters.quantization_type = static_cast<QuantizationType>(params["quantization_type"].asInt());

    return field;
}

Json::Value Schema::toJson() const {
    Json::Value json;
    for (const auto& field : fields) {
        json["fields"].append(field.toJson());
    }
    return json;
}

Schema Schema::fromJson(const Json::Value& json) {
    Schema schema;
    for (const auto& jsonField : json["fields"]) {
        schema.fields.push_back(Field::fromJson(jsonField));
    }
    return schema;
}
} // namespace lintdb