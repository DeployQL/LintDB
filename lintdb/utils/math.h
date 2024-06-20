#ifndef MATH_H
#define MATH_H

#include "mlas/mlas.h"

namespace lintdb {
std::vector<float> normalize_vector(const std::vector<float>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }

    // Calculate the L2 norm
    float norm = 0.0f;
    for (float value : input) {
        norm += value * value;
    }
    norm = std::sqrt(norm);

    if (norm == 0.0f) {
        throw std::runtime_error("Cannot normalize a zero-length vector");
    }

    // Normalize the vector
    std::vector<float> normalized(input.size());
    std::transform(input.begin(), input.end(), normalized.begin(), [norm](float value) {
        return value / norm;
    });

    return normalized;
}
}

#endif //MATH_H
