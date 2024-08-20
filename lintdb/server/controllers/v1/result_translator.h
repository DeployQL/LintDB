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
};
}