#pragma once

#include "lintdb/quantizers/Quantizer.h"

namespace lintdb {

class IdentityQuantizer : public Quantizer {
   public:
    IdentityQuantizer(size_t dim) : dim(dim) {}

    void train(const size_t n, const float* x, const size_t dim) override;

    void save(const std::string path) override;

    void sa_encode(size_t n, const float* x, residual_t* codes) override;

    void sa_decode(size_t n, const residual_t* codes, float* x) override;

    size_t code_size() override;

    size_t get_nbits() override;

    QuantizerType get_type() override;

   private:
    size_t dim;
};

} // namespace lintdb
