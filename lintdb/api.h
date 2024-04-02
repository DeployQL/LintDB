#ifndef LINTDB_API_H
#define LINTDB_API_H

#include <cstdint>

typedef int64_t idx_t;

// the codes used to save the centroid for each token vector.
// each code is treated as an index, which is defined above.
typedef idx_t
        code_t;
typedef uint8_t residual_t; // the residual codes saved for each token vector.

#endif