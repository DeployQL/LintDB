#ifndef LINTDB_BINARIZER_H
#define LINTDB_BINARIZER_H

#include <vector>
#include <string>
#include <stddef.h>
#include <memory>
#include "lintdb/api.h"

namespace lintdb {
    static const std::string BINARIZER_FILENAME = "_binarizer.bin";

    struct Binarizer {
        std::vector<float> bucket_cutoffs;
        std::vector<float> bucket_weights;
        float avg_residual;
        size_t nbits;
        size_t dim;
        std::vector<uint8_t> reverse_bitmap;
        std::vector<uint8_t> decompression_lut;

        Binarizer(size_t nbits, size_t dim);
        
        std::vector<uint8_t> binarize(const std::vector<float>& residuals);
        void train(size_t n, const float* x, size_t dim);
        void save(std::string path);

        void sa_encode(size_t n, const float* x, residual_t* codes);
        void sa_decode(size_t n, const residual_t* codes, float* x);

        static std::unique_ptr<Binarizer> load(std::string path);

        void calculate_quantiles(const std::vector<float>& heldoout_residual);


        std::vector<uint8_t> packbits(const std::vector<uint8_t>& binarized);
        std::vector<uint8_t> unpackbits(const std::vector<uint8_t>& packed, size_t dim, size_t nbits);
        // binarize takes in the residuals as floats, bucketizes them, and 
        // then returns the binarized version of the residuals.
        // the returned vector is of size dim * nbits.

        std::vector<uint8_t> create_reverse_bitmap();
        std::vector<uint8_t> create_decompression_lut();

    };
}

#endif