#ifndef LINTDB_BINARIZER_H
#define LINTDB_BINARIZER_H

#include <stdint.h>
#include <stddef.h>
#include <vector>

namespace lintdb {
    /**
     * Binarizer is a simple class that will encode and decode a float array into a uint8_t array.
    */
    struct Binarizer {
        void train(std::vector<float> data, size_t n);
        void encode(const float* data, uint8_t* codes, size_t n) const;
        void decode(const uint8_t* codes, float* data, size_t n) const;
    };
}

#endif