
#include <gtest/gtest.h>
#include <rocksdb/slice.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <vector>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/index.h"
#include "lintdb/invlists/RocksdbForwardIndex.h"
#include "lintdb/invlists/keys.h"

TEST(RocksDBTests, KeyEncodesAndDecodesCorrectly) {
    // this loop exists because we hit a decoding error, and I want to make sure
    // encoding/decoding works for more values.
    for (code_t i = 0; i < 20000; i++) {
        auto test_key = lintdb::Key{
                1, // tenant
                2, // inverted list id
                i  // doc id
        };

        std::string ks = test_key.serialize();
        auto slice = rocksdb::Slice(ks);

        auto decoded = lintdb::Key::from_slice(slice);

        EXPECT_EQ(test_key.tenant, decoded.tenant);
        EXPECT_EQ(test_key.inverted_list_id, decoded.inverted_list_id);
        EXPECT_EQ(test_key.id, decoded.id);
    }
}

TEST(RocksDBTests, TokenKeyEncodesAndDecodesCorrectly) {
    // this loop exists because we hit a decoding error, and I want to make sure
    // encoding/decoding works for more values.
    for (code_t i = 0; i < 20000; i++) {
        auto test_key = lintdb::TokenKey{
                1, // tenant
                2, // inverted list id
                i, // doc id
                3  // token id
        };

        std::string ks = test_key.serialize();
        auto slice = rocksdb::Slice(ks);

        auto decoded = lintdb::TokenKey::from_slice(slice);

        EXPECT_EQ(test_key.tenant, decoded.tenant);
        EXPECT_EQ(test_key.inverted_list_id, decoded.inverted_list_id);
        EXPECT_EQ(test_key.doc_id, decoded.doc_id);
        EXPECT_EQ(test_key.token_id, decoded.token_id);
    }
}

TEST(RocksDBTests, ForwardKeyEncodesAndDecodesCorrectly) {
    // this loop exists because we hit a decoding error, and I want to make sure
    // encoding/decoding works for more values.
    for (code_t i = 0; i < 20000; i++) {
        auto test_key = lintdb::ForwardIndexKey{
                1, // tenant
                i  // doc id
        };

        std::string ks = test_key.serialize();
        auto slice = rocksdb::Slice(ks);

        auto decoded = lintdb::ForwardIndexKey::from_slice(slice);

        EXPECT_EQ(test_key.tenant, decoded.tenant);
        EXPECT_EQ(test_key.id, decoded.id);
    }
}

TEST(RocksDBTests, Endianness) {
    std::vector<unsigned char> number;

    uint64_t num = 1;
    lintdb::store_bigendian(num, number);

    uint64_t decoded = lintdb::load_bigendian<uint64_t>(number.data());

    EXPECT_EQ(decoded, num);
}

TEST(RocksDBTests, ConcatNumbers) {
    std::vector<unsigned char> numbers;

    uint64_t num = 1;
    lintdb::store_bigendian<uint64_t>(num, numbers);

    uint32_t id = 2;
    lintdb::store_bigendian<uint32_t>(id, numbers);

    uint64_t decoded = lintdb::load_bigendian<uint64_t>(numbers.data());
    uint32_t decoded_id =
            lintdb::load_bigendian<uint32_t>(numbers.data() + sizeof(uint64_t));

    EXPECT_EQ(decoded, num);
    EXPECT_EQ(decoded_id, id);
}

TEST(RocksDBTests, MetadataSerialization) {
    std::map<std::string, std::string> metadata = {{"key", "metadata"}};

    std::vector<int64_t> codes(1, 1);
    std::vector<uint8_t> residuals(1, 2);
    lintdb::EncodedDocument test(codes, residuals, 1, 1, 0, metadata);

    std::string serialized = test.serialize_metadata();
    auto slice = rocksdb::Slice(serialized);

    auto st = slice.ToString();
    auto decoded = lintdb::DocumentMetadata::deserialize(st);

    EXPECT_EQ(metadata, decoded->metadata);
}