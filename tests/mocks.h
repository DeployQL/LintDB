#ifndef LINTDB_MOCKS_H
#define LINTDB_MOCKS_H

#include "lintdb/invlists/InvertedList.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/quantizers/ProductEncoder.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/invlists/IndexWriter.h"
#include "lintdb/Encoder.h"
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <gsl/span>

class MockInvertedList : public lintdb::InvertedList {
   public:
    MOCK_METHOD(void, add, (const uint64_t tenant, lintdb::EncodedDocument* doc), (override));
    MOCK_METHOD(void, remove, (const uint64_t tenant, std::vector<idx_t> ids), (override));
    MOCK_METHOD(void, merge, (rocksdb::DB* db, std::vector<rocksdb::ColumnFamilyHandle*>& cfs), (override));

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

    MOCK_METHOD(void, merge, (rocksdb::DB* db, std::vector<rocksdb::ColumnFamilyHandle*>& cfs), (override));
};

class MockIndexWriter : public lintdb::IndexWriter {
   public:
    MOCK_METHOD(void, write, (const BatchPostingData& batch_posting_data), (override));
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


#endif // LINTDB_MOCKS_H
