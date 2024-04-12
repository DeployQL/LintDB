
#include <gtest/gtest.h>
#include "lintdb/index.h"
#include "lintdb/EmbeddingBlock.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <faiss/utils/hamming.h>
#include <iostream>
#include "lintdb/invlists/RocksdbList.h"
#include "lintdb/invlists/util.h"
#include <rocksdb/slice.h>

TEST(RocksDBTests, KeyEncodesAndDecodesCorrectly) {
    // this loop exists because we hit a decoding error, and I want to make sure
    // encoding/decoding works for more values.
    for (code_t i=0; i < 20000; i++) {
        auto test_key = lintdb::Key{
            1, // tenant
            2, // inverted list id
            i // doc id
        };

        std::string ks = test_key.serialize();
        auto slice = rocksdb::Slice(ks);

        auto decoded = lintdb::Key::from_slice(slice);

        EXPECT_EQ(test_key.tenant, decoded.tenant);
        EXPECT_EQ(test_key.inverted_list_id, decoded.inverted_list_id);
        EXPECT_EQ(test_key.id, decoded.id);
    }

}

TEST(RocksDBTests, ForwardKeyEncodesAndDecodesCorrectly) {
    // this loop exists because we hit a decoding error, and I want to make sure
    // encoding/decoding works for more values.
    for (code_t i=0; i < 20000; i++) {
        auto test_key = lintdb::ForwardIndexKey{
            1, // tenant
            i // doc id
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
    uint32_t decoded_id = lintdb::load_bigendian<uint32_t>(numbers.data() + sizeof(uint64_t));

    EXPECT_EQ(decoded, num);
    EXPECT_EQ(decoded_id, id);
}

// TEST(RocksDBTests, InvertedListsWork) {
//     std::filesystem::path path = std::filesystem::temp_directory_path();
//     auto temp_db = path.append("test_index");
//     rocksdb::Options options;
//     options.create_if_missing = true;
//     options.create_missing_column_families = true;

//     std::vector<rocksdb::ColumnFamilyHandle*> column_family_handles;
//     auto cfs = lintdb::create_column_families();
//     rocksdb::DB* db;
//     std::string path = "ssomethign";
//     rocksdb::Status s = rocksdb::DB::Open(
//             options, path, cfs, &column_family_handles, &db);
//     assert(s.ok());

//     std::unique_ptr<lintdb::RocksDBInvertedList> ivf(new lintdb::RocksDBInvertedList(*db, column_family_handles));

//     for(int i=0; i<100; i++) {

//     }
//     ivf->add(std::make_unique<lintdb::EncodedDocument>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
// }