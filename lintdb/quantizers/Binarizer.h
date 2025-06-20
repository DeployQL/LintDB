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

    Binarizer(
            const std::vector<float>& bucket_cutoffs,
            const std::vector<float>& bucket_weights,
            const float avg_residual,
            const size_t nbits,
            const size_t dim);

    // copy constructor
    Binarizer(const Binarizer& other);

    Binarizer& operator=(Binarizer other) {
        swap(*this, other);
        return *this;
    }

    std::vector<uint8_t> binarize(const std::vector<float>& residuals);
    void train(const size_t n, const float* x, const size_t dim) override;
    void save(const std::string path) override;

    void sa_encode(size_t n, const float* x, residual_t* codes) override;
    void sa_decode(size_t n, const residual_t* codes, float* x) override;

    void sa_decode_generic (size_t n, const residual_t* codes, float* x);
#ifdef __AVX2__
    void sa_decode_1bit(size_t n, const residual_t* codes, float* x);
#endif

    size_t code_size() override;

    size_t get_nbits() override {
        return nbits;
    }

    static std::unique_ptr<Binarizer> load(std::string path);

    QuantizerType get_type() override;

    friend void swap(Binarizer& first, Binarizer& second) {
        std::swap(first.bucket_cutoffs, second.bucket_cutoffs);
        std::swap(first.bucket_weights, second.bucket_weights);
        std::swap(first.avg_residual, second.avg_residual);
        std::swap(first.nbits, second.nbits);
        std::swap(first.dim, second.dim);
        std::swap(first.reverse_bitmap, second.reverse_bitmap);
        std::swap(first.decompression_lut, second.decompression_lut);
    }

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