#pragma once

#include "lintdb/schema/DataTypes.h"

namespace lintdb {
    /**
 * ColumnInverter is a helper class that inverts a column of a document.
 *
 * Once the document processor has assigned codes to a tensor, we can invert it into the index.
 *
 * inverted index:
 *      key => tenant, field, IVF centroid id, doc_id
 *      value => codes assigned to this centroid
 * context index:
 *      key => tenant, field, doc_id
 *      value => values of the field
 * forward index:
 *      key => tenant, doc_id
 *      value => all stored data of the document
 */
    struct ProcessedData {
        uint64_t tenant;
        uint8_t field;
        std::vector<idx_t> centroid_ids;
        idx_t doc_id;

        FieldValue value;
    };
}