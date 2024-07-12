#include "IdentityQuantizer.h"

namespace lintdb {
    void IdentityQuantizer::train(const size_t n, const float *x, const size_t dim) {

    }

    void IdentityQuantizer::save(const std::string path) {
           }

    void IdentityQuantizer::sa_encode(size_t n, const float *x, residual_t *codes) {
        codes = (residual_t *) x;
    }

    void IdentityQuantizer::sa_decode(size_t n, const residual_t *codes, float *x) {
        x = (float *) codes;
    }

    size_t IdentityQuantizer::code_size() {
        return dim * sizeof(float);
    }

    size_t IdentityQuantizer::get_nbits() {
        return dim * sizeof(float);
    }

    QuantizerType IdentityQuantizer::get_type() {
        return NONE;
    }
} // lintdb