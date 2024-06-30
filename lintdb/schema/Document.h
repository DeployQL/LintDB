#pragma once

#include <unordered_map>
#include "DataTypes.h"

namespace lintdb {
struct Document {
    std::unordered_map<std::string, FieldValue> fields;

    void addField(const std::string& name, const FieldValue& value) {
        fields[name] = value;
    }
};
}


