#pragma once

#include <memory>
#include <json/reader.h>
#include <json/writer.h>
#include "lintdb/SearchResult.h"
#include "lintdb/schema/DataTypes.h"

namespace server {
class SearchResultJsonTranslator {
   public:
    static Json::Value toJson(const lintdb::SearchResult& result) {
        Json::Value root;
        root["id"] = static_cast<Json::Int64>(result.id);
        root["score"] = result.score;

        Json::Value metadataJson;
        for (const auto& [key, value] : result.metadata) {

            metadataJson[key] = lintdb::supportedTypeToJSON(value);
        }
        root["metadata"] = metadataJson;

        return root;
    }

    static lintdb::SearchResult fromJson(const Json::Value& json) {
        lintdb::SearchResult result;
        result.id = json["id"].asInt64();
        result.score = json["score"].asFloat();

        const Json::Value& metadataJson = json["metadata"];
        for (const auto& key : metadataJson.getMemberNames()) {
            result.metadata[key] = lintdb::jsonToSupportedType(metadataJson[key]);
        }

        return result;
    }
};
}