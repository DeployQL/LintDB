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
    std::unordered_map<std::string, FieldValue> fields;
    idx_t id; /// the unique id of the document

    void addField(const std::string& name, const FieldValue& value) {
        fields[name] = value;
    }
};

}


