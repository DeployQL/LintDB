#pragma once

#include <unordered_map>
#include "DataTypes.h"

namespace lintdb {
/**
 * Documents hold data as they are passed into the database from the user.
 *
 * Each Document must have a unique id.
 */
struct Document {
    std::vector<FieldValue> fields;
    idx_t id; /// the unique id of the document

    Document(idx_t id, const std::vector<FieldValue>& fields)
            : fields(fields), id(id) {}

    Json::Value toJson() const {
        Json::Value root;
        root["id"] = static_cast<Json::Int64>(id);

        Json::Value fieldsArray(Json::arrayValue);
        for (const auto &field : fields) {
            fieldsArray.append(field.toJson());
        }
        root["fields"] = fieldsArray;

        return root;
    }

    static Document fromJson(const Json::Value &json) {
        idx_t id = json["id"].asInt64();

        std::vector<FieldValue> fields;
        const Json::Value &fieldsArray = json["fields"];
        for (const auto &fieldJson : fieldsArray) {
            fields.push_back(FieldValue::fromJson(fieldJson));
        }

        return Document(id, fields);
    }
};

} // namespace lintdb
