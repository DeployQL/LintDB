
#include <gtest/gtest.h>
#include "lintdb/index.h"
#include "lintdb/EmbeddingBlock.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>
#include "lintdb/plaid.h"
#include <filesystem>
#include <faiss/utils/hamming.h>
#include <iostream>
#include <arrayfire.h>
#include "lintdb/invlists/RocksdbList.h"
#include "lintdb/invlists/util.h"
#include <rocksdb/slice.h>

TEST(RocksDBTests, KeyEncodesAndDecodesCorrectly) {
    auto test_key = lintdb::Key{
        1, // tenant
        2, // inverted list id
        3 // doc id
    };

    std::string ks = test_key.serialize();
    auto slice = rocksdb::Slice(ks);

    auto decoded = lintdb::Key::from_slice(slice);

    EXPECT_EQ(test_key.tenant, decoded.tenant);
    EXPECT_EQ(test_key.inverted_list_id, decoded.inverted_list_id);
    EXPECT_EQ(test_key.id, decoded.id);
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
    uint32_t decoded_id = lintdb::load_bigendian<uint32_t>(numbers.data() + sizeof(uint64_t));

    EXPECT_EQ(decoded, num);
    EXPECT_EQ(decoded_id, id);
}