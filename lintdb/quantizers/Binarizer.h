#ifndef LINTDB_BINARIZER_H
#define LINTDB_BINARIZER_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "lintdb/api.h"
#include "lintdb/quantizers/PQDistanceTables.h"
#include "lintdb/quantizers/Quantizer.h"

namespace lintdb {
struct Binarizer : public Quantizer {
    std::vector<float> bucket_cutoffs;
    std::vector<float> bucket_weights;
    float avg_residual;
    size_t nbits;
    size_t dim;
    std::vector<uint8_t> reverse_bitmap;
    std::vector<uint8_t> decompression_lut;

    Binarizer(size_t nbits, size_t dim);

    std::vector<uint8_t> binarize(const std::vector<float>& residuals);
    void train(const size_t n, const float* x, const size_t dim) override;
    void save(const std::string path) override;

    void sa_encode(size_t n, const float* x, residual_t* codes) override;
    void sa_decode(size_t n, const residual_t* codes, float* x) override;
    size_t code_size() override;

    size_t get_nbits() override {
        return nbits;
    }

    static std::unique_ptr<Binarizer> load(std::string path);

    void set_weights(
            const std::vector<float>& weights,
            const std::vector<float>& cutoffs,
            const float avg_residual);

    QuantizerType get_type() override;

   private:
    void calculate_quantiles(const std::vector<float>& heldoout_residual);

    std::vector<uint8_t> bucketize(const std::vector<float>& residuals);
    std::vector<uint8_t> packbits(const std::vector<uint8_t>& binarized);
    std::vector<uint8_t> unpackbits(
            const std::vector<uint8_t>& packed,
            size_t dim,
            size_t nbits);
    // binarize takes in the residuals as floats, bucketizes them, and
    // then returns the binarized version of the residuals.
    // the returned vector is of size dim * nbits.

    std::vector<uint8_t> create_reverse_bitmap();
    std::vector<uint8_t> create_decompression_lut();
};
} // namespace lintdb

#endif