
#include <gtest/gtest.h>
#include <vector>
#define private public
#include "lintdb/Binarizer.h"
#include <cmath>
#include <glog/logging.h>


TEST(BinarizerTests, QuantilesTest) {
    // Example test data
    std::vector<float> heldout_avg_residual = {1.0, 2.0, 3.0, 4.0, 5.0};
    int nbits = 4;

    lintdb::Binarizer binarizer(nbits, 128);
    // Call the function to calculate bucket weights
    binarizer.calculate_quantiles(heldout_avg_residual);

    // Check the size of the bucket weights
    ASSERT_EQ(binarizer.bucket_weights.size(), 16); // In this example, nbits = 4, so 2^4 - 1 = 15
}

TEST(BinarizerTests, PackBitsTests) {
    // input is dim * nbits. we assume we've already binarized the data.
    std::vector<uint8_t> input = {1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0}; // Example input data
    size_t dim = 16;
    size_t nbits = 2;

    lintdb::Binarizer binarizer(nbits, dim);

    // Test packing and unpacking
    std::vector<uint8_t> packed = binarizer.packbits(input);
    ASSERT_EQ(packed.size(), dim / 8 * nbits); // Expected size of packed data (2 bits per value, 16 values = 16 / 8 * 2 = 4 bytes)
    
    std::vector<uint8_t> unpacked = binarizer.unpackbits(packed, dim, nbits);
    ASSERT_EQ(unpacked.size(), dim*nbits); // Expected size of unpacked data
    // Check if the unpacked data matches the input data
    ASSERT_EQ(input, unpacked);
}

TEST(BinarizerTests, PackBitsTestLargeInput) {
    size_t dim = 128;
    size_t nbits = 2;
    std::vector<uint8_t> input(128*nbits, 1); // All bits set to 1

    lintdb::Binarizer binarizer(nbits, dim);

    // Test packing and unpacking
    std::vector<uint8_t> packed = binarizer.packbits(input);
    std::vector<uint8_t> unpacked = binarizer.unpackbits(packed, dim, nbits);

    // Check if the unpacked data matches the input data
    ASSERT_EQ(input, unpacked);
}

TEST(BinarizerTests, DecompressionLUT) {
    // Create a mock Binarizer instance for testing
    std::vector<float> bucket_weights = {0.1, 0.2, 0.3}; // Example bucket weights
    size_t nbits = 2; // Example number of bits
    lintdb::Binarizer binarizer(nbits, 16); // dim isn't necessary here, but we must pass it to init.
    binarizer.bucket_weights = bucket_weights;

    // Generate the decompression lookup table
    std::vector<uint8_t> lut = binarizer.create_decompression_lut();

    size_t keys_per_byte = 8 / nbits;
    size_t num_keys = bucket_weights.size();
    size_t num_entries = pow(num_keys, keys_per_byte);

    // Perform assertions on the generated lookup table
    ASSERT_EQ(lut.size(), num_entries * keys_per_byte); // Expected size of lookup table for 3 bucket weights and 2 bits per lookup
}

TEST(BinarizerTests, EncodingTest) {
    // we get the number of buckets by 1 << nbits. 1 << 2 == 4
    size_t nbits = 2; // Example number of bits
    lintdb::Binarizer binarizer(nbits, 16); // harcoding dim as 8.

    std::vector<float> input = {0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2}; // Example input data
    binarizer.train(1, input.data(), 16);

    for (auto cutoff : binarizer.bucket_cutoffs) {
        LOG(INFO) << "bucket cutoff: " << cutoff;
    }
    for (auto bucket : binarizer.bucket_weights) {
        LOG(INFO) << "bucket weight: " << bucket;
    }


    std::vector<uint8_t> output(16 / 8 * nbits, 0); // 
    binarizer.sa_encode(1, input.data(), output.data());

    std::vector<float> decoded(16, 0);
    binarizer.sa_decode(1, output.data(), decoded.data());

    ASSERT_EQ(input, decoded);
    

}