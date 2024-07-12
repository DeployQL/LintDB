#pragma once

#include <vector>
#include "lintdb/schema/DataTypes.h"
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
    };

}