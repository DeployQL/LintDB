#include "Schema.h"

namespace lintdb {
Json::Value Field::toJson() const {
    Json::Value json;
    json["name"] = name;
    json["data_type"] = static_cast<int>(data_type);

    Json::Value fieldTypesJson(Json::arrayValue);
    for (const auto& fieldType : field_types) {
        fieldTypesJson.append(static_cast<int>(fieldType));
    }
    json["field_types"] = fieldTypesJson;

    Json::Value params;
    params["dimensions"] = parameters.dimensions;
    params["analyzer"] = parameters.analyzer;
    params["quantization"] = static_cast<int>(parameters.quantization);
    params["num_centroids"] = parameters.num_centroids;
    params["num_iterations"] = parameters.num_iterations;
    params["num_subquantizers"] = parameters.num_subquantizers;
    params["nbits"] = parameters.nbits;
    json["parameters"] = params;

    return json;
}

Field Field::fromJson(const Json::Value& json) {
    Field field;
    field.name = json["name"].asString();
    field.data_type = static_cast<DataType>(json["data_type"].asInt());

    const Json::Value& fieldTypesJson = json["field_types"];
    for (const auto& fieldTypeJson : fieldTypesJson) {
        field.field_types.push_back(static_cast<FieldType>(fieldTypeJson.asInt()));
    }

    const Json::Value& params = json["parameters"];
    field.parameters.dimensions = params["dimensions"].asUInt();
    field.parameters.analyzer = params["analyzer"].asString();
    field.parameters.quantization = static_cast<QuantizerType>(params["quantization"].asInt());
    field.parameters.num_centroids = params["num_centroids"].asUInt();
    field.parameters.num_iterations = params["num_iterations"].asUInt();
    field.parameters.num_subquantizers = params["num_subquantizers"].asUInt();
    field.parameters.nbits = params["nbits"].asUInt();

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