//
//

#ifndef LINTDB_MOCKS_H
#define LINTDB_MOCKS_H

#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/quantizers/ProductEncoder.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/Encoder.h"
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <gsl/span>

class MockInvertedList : public lintdb::InvertedList {
   public:
    MOCK_METHOD(void, add, (const uint64_t tenant, lintdb::EncodedDocument* doc), (override));
    MOCK_METHOD(void, remove, (const uint64_t tenant, std::vector<idx_t> ids), (override));
    MOCK_METHOD(void, merge, (rocksdb::DB* db, std::vector<rocksdb::ColumnFamilyHandle*> cfs), (override));

    MOCK_METHOD(std::unique_ptr<lintdb::Iterator>, get_iterator, (const uint64_t tenant, const idx_t inverted_list), (const, override));
    MOCK_METHOD(std::vector<idx_t>, get_mapping, (const uint64_t tenant, idx_t id), (const, override));
};

class MockForwardIndex : public lintdb::ForwardIndex {
   public:
    MOCK_METHOD(std::vector<std::unique_ptr<lintdb::DocumentCodes>>, get_codes,
                (const uint64_t tenant, const std::vector<idx_t>& ids), (const, override));
    MOCK_METHOD(std::vector<std::unique_ptr<lintdb::DocumentResiduals>>, get_residuals,
                (const uint64_t tenant, const std::vector<idx_t>& ids), (const, override));
    MOCK_METHOD(std::vector<std::unique_ptr<lintdb::DocumentMetadata>>, get_metadata,
                (const uint64_t tenant, const std::vector<idx_t>& ids), (const, override));

    MOCK_METHOD(void, add, (const uint64_t tenant, lintdb::EncodedDocument* doc, bool store_codes), (override));
    MOCK_METHOD(void, remove, (const uint64_t tenant, std::vector<idx_t> ids), (override));

    MOCK_METHOD(void, merge, (rocksdb::DB* db, std::vector<rocksdb::ColumnFamilyHandle*> cfs), (override));
};

class MockEncoder : public lintdb::Encoder {
   public:
    MOCK_METHOD(size_t, get_dim, (), (const, override));
    MOCK_METHOD(size_t, get_num_centroids, (), (const, override));
    MOCK_METHOD(size_t, get_nbits, (), (const, override));

    MOCK_METHOD(std::unique_ptr<lintdb::EncodedDocument>, encode_vectors, (const lintdb::EmbeddingPassage& doc), (override));

    MOCK_METHOD(std::vector<float>, decode_vectors, (
                                                            const gsl::span<const code_t> codes,
                                                            const gsl::span<const residual_t> residuals,
                                                            const size_t num_tokens), (const, override));

    MOCK_METHOD(void, search, (
                                      const float* data,
                                      const int n,
                                      std::vector<idx_t>& coarse_idx,
                                      std::vector<float>& distances,
                                      const size_t k_top_centroids,
                                      const float centroid_threshold), (override));

    MOCK_METHOD(void, search_quantizer, (
                                                const float* data,
                                                const int num_query_tok,
                                                std::vector<idx_t>& coarse_idx,
                                                std::vector<float>& distances,
                                                const size_t k_top_centroids,
                                                const float centroid_threshold), (override));

    MOCK_METHOD(float*, get_centroids, (), (const, override));
    MOCK_METHOD(lintdb::Quantizer*, get_quantizer, (), (const, override));

    MOCK_METHOD(void, save, (std::string path), (override));

    MOCK_METHOD(void, train, (
                                     const float* embeddings,
                                     const size_t n,
                                     const size_t dim,
                                     const int n_list,
                                     const int n_iter), (override));

    MOCK_METHOD(void, set_centroids, (float* data, int n, int dim), (override));

    MOCK_METHOD(void, set_weights, (
                                           const std::vector<float>& weights,
                                           const std::vector<float>& cutoffs,
                                           const float avg_residual), (override));
};

class MockProductEncoder : public lintdb::ProductEncoder {
   public:
    MockProductEncoder(size_t dim, size_t nbits, size_t num_subquantizers)
            : lintdb::ProductEncoder(dim, nbits, num_subquantizers) {}

    MOCK_METHOD(void, sa_encode, (size_t n, const float* x, residual_t* codes), (override));
    MOCK_METHOD(void, sa_decode, (size_t n, const residual_t* codes, float* x), (override));
    MOCK_METHOD(size_t, code_size, (), (override));
    MOCK_METHOD(size_t, get_nbits, (), (override));
    MOCK_METHOD(void, save, (const std::string path), (override));
    MOCK_METHOD(void, train, (const size_t n, const float* embeddings, const size_t dim), (override));
    MOCK_METHOD(lintdb::QuantizerType, get_type, (), (override));

    // Mock the static method `load`
    MOCK_METHOD(std::unique_ptr<lintdb::ProductEncoder>, load, (std::string path, lintdb::QuantizerConfig& config));
};

#endif // LINTDB_MOCKS_H
