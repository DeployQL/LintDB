#ifndef LINTDB_UTIL_H
#define LINTDB_UTIL_H

#include <stdint.h>
#include <stddef.h>

namespace lintdb {
    /**
 * Normalize vector normalizes vectors in place.
 * 
 * do i need to consider simd instructions for optimizations?
 * https://stackoverflow.com/questions/57469359/how-to-efficiently-normalize-vector-c
*/
void normalize_vector(float* doc_residuals, const size_t num_doc_tokens, const size_t dim);
}

#endif