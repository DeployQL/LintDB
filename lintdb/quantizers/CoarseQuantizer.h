#ifndef LINTDB_COARSEQUANTIZER_H
#define LINTDB_COARSEQUANTIZER_H

#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include "lintdb/quantizers/impl/kmeans.h"
#include "lintdb/quantizers/Quantizer.h"
#include <gsl/span>
#include <memory>
#include "lintdb/version.h"

namespace lintdb {

/**
 * @class CoarseQuantizer
 * @brief This class is used for quantization of vectors.
 *
 * It provides methods for training, saving, assigning, decoding, computing residuals, reconstructing, searching, resetting, adding, getting type, getting xb, serializing and deserializing.
 */
class CoarseQuantizer {
   public:
    bool is_trained; ///< Indicates if the quantizer is trained or not

    /**
     * @brief Constructor that initializes the dimensionality of input vectors.
     * @param d Dimensionality of input vectors
     */
    CoarseQuantizer(size_t d);

    /**
     * @brief Trains the quantizer.
     * @param n Number of vectors
     * @param x Pointer to the vectors
     * @param k Number of centroids
     * @param num_iter Number of iterations for training, default is 10
     */
    void train(const size_t n, const float* x, size_t k, size_t num_iter=10);

    /**
     * @brief Saves the quantizer to a file.
     * @param path Path to the file
     */
    void save(const std::string& path);

    /**
     * @brief Assigns codes to vectors.
     * @param n Number of vectors
     * @param x Pointer to the vectors
     * @param codes Pointer to the codes
     */
    void assign(size_t n, const float* x, idx_t* codes);

    /**
     * @brief Decodes the codes to vectors.
     * @param n Number of codes
     * @param codes Pointer to the codes
     * @param x Pointer to the vectors
     */
    void sa_decode(size_t n, const idx_t* codes, float* x);

    /**
     * @brief Computes the residual of a vector.
     * @param vec Pointer to the vector
     * @param residual Pointer to the residual
     * @param centroid_id ID of the centroid
     */