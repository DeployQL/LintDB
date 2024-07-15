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
        FieldMapper(const FieldMapper& other) {
            nameToID = other.nameToID;
            fieldID = other.fieldID;
            idToField = other.idToField;
        }

        // copy assignment operator
        // using copy and swap idiom.
        FieldMapper &operator=(FieldMapper other) {
            std::swap(nameToID, other.nameToID);
            std::swap(fieldID, other.fieldID);
            std::swap(idToField, other.idToField);
            return *this;
        }

        inline void addSchema(const Schema &schema) {
            for (const auto &field: schema.fields) {
                addMapping(field);
            }
        }

        inline DataType getDataType(const uint8_t field_id) const {
            return idToField.at(field_id).data_type;
        }

        inline std::vector<FieldType> getFieldTypes(const uint8_t field_id) const {
            return idToField.at(field_id).field_types;
        }

        inline int getFieldID(const std::string &fieldName) const {
            auto it = nameToID.find(fieldName);
            if (it != nameToID.end()) {
                return it->second;
            }
            throw std::runtime_error("Field name not found: " + fieldName);
        }

        inline std::string getFieldName(int fieldID) const {
            auto it = idToField.find(fieldID);
            if (it != idToField.end()) {
                return it->second.name;
            }
            throw std::runtime_error("Field ID not found");
        }

        inline Json::Value toJson() const {
            Json::Value json;
            for (const auto &pair: nameToID) {
                json["nameToID"][pair.first] = pair.second;
            }
            for(const auto &pair: idToField) {
                json["idToField"][pair.first] = pair.second.toJson();
            }
            return json;
        }

        static std::shared_ptr<FieldMapper> fromJson(const Json::Value &json);

    private:
        std::unordered_map<int, Field> idToField;
        std::unordered_map<std::string, int> nameToID;
        int fieldID = 0;

        inline void addMapping(const Field& field) {
            if (nameToID.find(field.name) != nameToID.end()) {
                throw std::runtime_error("Field name already exists: " + field.name);
            }
            nameToID[field.name] = fieldID;
            idToField[fieldID] = field;

            fieldID++;
        }

    };
}