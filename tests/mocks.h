#ifndef LINTDB_MOCKS_H
#define LINTDB_MOCKS_H

#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/quantizers/ProductEncoder.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/invlists/IndexWriter.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <gsl/span>


class MockIndexWriter : public lintdb::IIndexWriter {
   public:
    MOCK_METHOD(void, write, (const lintdb::BatchPostingData& batch_posting_data), (override));
};

class MockQuantizer : public lintdb::Quantizer {
   public:
    MOCK_METHOD(void, train, (const size_t n, const float* x, const size_t dim), (override));
    MOCK_METHOD(void, save, (const std::string path), (override));
    MOCK_METHOD(void, sa_encode, (size_t n, const float* x, residual_t* codes), (override));
    MOCK_METHOD(void, sa_decode, (size_t n, const residual_t* codes, float* x), (override));
    MOCK_METHOD(size_t, code_size, (), (override));
    MOCK_METHOD(size_t, get_nbits, (), (override));
    MOCK_METHOD(lintdb::QuantizerType, get_type, (), (override));
};

class MockCoarseQuantizer : public lintdb::ICoarseQuantizer {
public:
    MOCK_METHOD(void, train, (const size_t n, const float* x, size_t k, size_t num_iter), (override));
    MOCK_METHOD(void, save, (const std::string& path), (override));
    MOCK_METHOD(void, assign, (size_t n, const float* x, idx_t* codes), (override));
    MOCK_METHOD(void, sa_decode, (size_t n, const idx_t* codes, float* x), (override));
    MOCK_METHOD(void, compute_residual, (const float* vec, float* residual, idx_t centroid_id), (override));
    MOCK_METHOD(void, compute_residual_n, (int n, const float* vec, float* residual, idx_t* centroid_ids), (override));
    MOCK_METHOD(void, reconstruct, (idx_t centroid_id, float* embedding), (override));
    MOCK_METHOD(void, search, (size_t num_query_tok, const float* data, size_t k_top_centroids, float* distances, idx_t* coarse_idx), (override));
    MOCK_METHOD(void, reset, (), (override));
    MOCK_METHOD(void, add, (int n, float* data), (override));
    MOCK_METHOD(size_t, code_size, (), (override));
    MOCK_METHOD(size_t, num_centroids, (), (override));
    MOCK_METHOD(float*, get_xb, (), (override));
    MOCK_METHOD(void, serialize, (const std::string& filename), (const, override));
    MOCK_METHOD(bool, is_trained, (), (const, override));

};

#endif // LINTDB_MOCKS_H
