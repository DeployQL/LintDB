#pragma once

#include "DocValue.h"
#include "QueryContext.h"

namespace lintdb {
    /**
     * decode_vectors manages the decoding of vectors from the doc_value. All tensors become
     * QuantizedTensor values going into the index, and we need to decode any tensors that have
     * an associated quantizer.
     *
     * @param context
     * @param doc_value
     * @return
     */
    DocValue decode_vectors(const QueryContext& context, const DocValue& doc_value);
}