#include "FieldMapper.h"
#include <json/json.h>

namespace lintdb {
    std::shared_ptr<FieldMapper> FieldMapper::fromJson(const Json::Value &json) {
        std::shared_ptr<FieldMapper> mapper = std::make_shared<FieldMapper>();
        int highest_id = 0;
        for (const auto &member: json["nameToID"].getMemberNames()) {
            mapper->nameToID[member] = json["nameToID"][member].asInt();
            if (mapper->nameToID[member] > highest_id) {
                highest_id = mapper->nameToID[member];
            }
        }
        for (const auto &member: json["idToName"].getMemberNames()) {
            mapper->idToName[std::stoi(member)] = json["idToName"][member].asString();
        }
        for (const auto &member: json["idToType"].getMemberNames()) {
            // map the int key to the enum type
            mapper->idToDataType[std::stoi(member)] = IntToDataType.at(json["idToType"][member].asInt());
        }
        mapper->fieldID = highest_id + 1;
        return mapper;
    }
}