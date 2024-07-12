#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <json/json.h>
#include "lintdb/schema/Schema.h"
#include "lintdb/schema/DataTypes.h"
#include <glog/logging.h>

namespace lintdb {

    class FieldMapper {
    public:
        FieldMapper() = default;
        //copy constructor
        FieldMapper(const FieldMapper &other) {
            nameToID = other.nameToID;
            idToName = other.idToName;
            idToDataType = other.idToDataType;
            fieldID = other.fieldID;
        }

        // copy assignment operator
        // using copy and swap idiom.
        FieldMapper &operator=(FieldMapper other) {
            std::swap(nameToID, other.nameToID);
            std::swap(idToName, other.idToName);
            std::swap(idToDataType, other.idToDataType);
            std::swap(fieldID, other.fieldID);
            return *this;
        }

        inline void addSchema(const Schema &schema) {
            for (const auto &field: schema.fields) {
                addMapping(field);
            }
        }

        inline DataType getDataType(const uint8_t field_id) const {
            return idToDataType.at(field_id);
        }

        inline int getFieldID(const std::string &fieldName) const {
            auto it = nameToID.find(fieldName);
            if (it != nameToID.end()) {
                return it->second;
            }
            throw std::runtime_error("Field name not found: " + fieldName);
        }

        inline std::string getFieldName(int fieldID) const {
            auto it = idToName.find(fieldID);
            if (it != idToName.end()) {
                return it->second;
            }
            throw std::runtime_error("Field ID not found");
        }

        inline Json::Value toJson() const {
            Json::Value json;
            for (const auto &pair: nameToID) {
                json["nameToID"][pair.first] = pair.second;
            }
            for (const auto &pair: idToName) {
                json["idToName"][std::to_string(pair.first)] = pair.second;
            }
            for (const auto &pair: idToDataType) {
                json["idToDataType"][std::to_string(pair.first)] = DataTypeToInt.at(pair.second);
            }
            return json;
        }

        static std::shared_ptr<FieldMapper> fromJson(const Json::Value &json);

    private:
        std::unordered_map<std::string, int> nameToID;
        std::unordered_map<int, std::string> idToName;
        std::unordered_map<int, DataType> idToDataType;
        int fieldID = 0;

        inline void addMapping(const Field& field) {
            if (nameToID.find(field.name) != nameToID.end()) {
                throw std::runtime_error("Field name already exists: " + field.name);
            }
            nameToID[field.name] = fieldID;
            idToName[fieldID] = field.name;
            idToDataType[fieldID] = field.data_type;

            fieldID++;
        }

    };
}