#pragma once

#include <vector>
#include "lintdb/api.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/schema/DataTypes.h"

namespace lintdb {
/**
 * DocValue is a simple struct that holds a field value and the field id.
 *
 * It is the job of the caller to ensure that the field is valid, because this
 * class has no concept of what the field should look like.
 */
struct DocValue {
    lintdb::SupportedTypes value;
    uint8_t field_id;
    DataType type;
    bool unread_value =
            false; /// ColBERT fields do not have their values decoded from the
                   /// index. We check this flag so that
    /// we can throw an exception if the user tries to access the value.

    DocValue(SupportedTypes value, uint8_t field_id, DataType type)
            : value(std::move(value)), field_id(field_id), type(type) {}

    SupportedTypes get_value() const {
        if (unread_value) {
            throw LintDBException(
                    "Document's value was not decoded from the index. This is likely because a ColBERT field was read");
        }
        return value;
    }
};

} // namespace lintdb