#pragma once

#include "lintdb/query/QueryNode.h"
#include "lintdb/schema/DataTypes.h"
#include <memory>
#include <json/reader.h>
#include <json/writer.h>

namespace server {
class QueryNodeJsonTranslator {
   public:
    static std::unique_ptr<lintdb::QueryNode> fromJson(const Json::Value& json) {
        std::string type_string = json["type"].asString();

        lintdb::QueryNodeType type;
        if(type_string == "TERM") {
            type = lintdb::QueryNodeType::TERM;
        } else if (type_string == "TENSOR") {
            type = lintdb::QueryNodeType::VECTOR;
        } else if (type_string == "AND") {
            type = lintdb::QueryNodeType::AND;
        } else {
            throw std::runtime_error("unknown QueryNodeType");
        }

        switch (type) {
            case lintdb::QueryNodeType::TERM: {
                lintdb::FieldValue value = lintdb::FieldValue::fromJson(json["value"]);
                return std::make_unique<lintdb::TermQueryNode>(value);
            }
            case lintdb::QueryNodeType::VECTOR: {
                lintdb::Tensor value;
                for(auto& v : json["value"]) {
                    value.push_back(v.asFloat());
                }
                uint64_t num_tensors = json["num_tensors"].asUInt64();
                std::string field = json["name"].asString();
                lintdb::FieldValue fv = lintdb::FieldValue(field, value, size_t(num_tensors));
                return std::make_unique<lintdb::VectorQueryNode>(fv);
            }
            case lintdb::QueryNodeType::AND: {
                std::vector<std::unique_ptr<lintdb::QueryNode>> children;
                for (const auto& childJson : json["children"]) {
                    children.push_back(fromJson(childJson));
                }
                return std::make_unique<lintdb::AndQueryNode>(std::move(children));
            }
            default:
                throw std::runtime_error("Unknown QueryNodeType");
        }
    }
};
}