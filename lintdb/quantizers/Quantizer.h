#pragma once

#include <stddef.h>
#include <string>
#include "lintdb/api.h"
#include "lintdb/quantizers/PQDistanceTables.h"

namespace lintdb {
static const std::string QUANTIZER_FILENAME = "_residual_quantizer.bin";
static const std::string LEGACY_QUANTIZER_FILENAME = "_binarizer.bin";

enum QuantizerType {
    UNKNOWN,
    NONE,
    BINARIZER,
    PRODUCT_ENCODER,
};

struct QuantizerConfig {
    size_t nbits;
    size_t dim;
    size_t num_subquantizers; // used in ProductEncoder
};
/**
 * Quantizer is responsible for vector encoding. Unlike the Encoder, this isn't
 * responsible for IVF assignment.
 */
struct Quantizer {
    virtual void train(const size_t n, const float* x, const size_t dim) = 0;
    virtual void save(const std::string path) = 0;

    virtual void sa_encode(size_t n, const float* x, residual_t* codes) = 0;
    virtual void sa_decode(size_t n, const residual_t* codes, float* x) = 0;
    virtual size_t code_size() = 0;

    virtual size_t get_nbits() = 0;

    virtual QuantizerType get_type() = 0;

    virtual ~Quantizer() = default;
};
} // namespace lintdb