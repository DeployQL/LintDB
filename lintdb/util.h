#ifndef LINTDB_UTIL_H
#define LINTDB_UTIL_H

#include <json/reader.h>
#include <json/writer.h>
#include <stddef.h>
#include <stdint.h>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include "lintdb/SearchOptions.h"

namespace lintdb {
/**
 * Normalize vector normalizes vectors in place.
 *
 * do i need to consider simd instructions for optimizations?
 * https://stackoverflow.com/questions/57469359/how-to-efficiently-normalize-vector-c
 */
void normalize_vector(
        float* doc_residuals,
        const size_t num_doc_tokens,
        const size_t dim);

template <typename T>
void product_helper(
        const std::vector<std::vector<T>>& pools,
        std::vector<T>& result,
        size_t index,
        std::vector<T>& current) {
    if (index == pools.size()) {
        for (const auto& elem : current) {
            result.push_back(elem);
        }
        return;
    }
    for (const auto& element : pools[index]) {
        current.push_back(element);
        product_helper(pools, result, index + 1, current);
        current.pop_back();
    }
}

/**
 * product creates the cartesian product of a range of elements. Similar to
 * python, it enables us to repeat the input a certain amount of times.
 */
template <typename T>
std::vector<T> product(
        const std::vector<std::vector<T>>& args,
        size_t repeat = 1) {
    std::vector<std::vector<T>> pools;
    for (const auto& arg : args) {
        pools.insert(pools.end(), repeat, arg);
    }
    std::vector<T> result;
    std::vector<T> current;
    product_helper(pools, result, 0, current);
    return result;
}

Json::Value loadJson(const std::string& path);

inline std::vector<size_t> subsample(const size_t total, const size_t sample) {
    std::mt19937 rng;
    std::seed_seq seed{1234};

    rng.seed(seed);

    std::uniform_int_distribution<size_t> dist(0, total - 1);
    std::vector<size_t> indices;
    for (size_t i = 0; i < sample; i++) {
        indices.push_back(dist(rng));
    }

    return indices;
}

} // namespace lintdb

#endif