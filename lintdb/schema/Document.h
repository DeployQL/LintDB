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
};

} // namespace lintdb
