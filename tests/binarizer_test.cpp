
#include <gtest/gtest.h>
#include <vector>
#define private public
#include "lintdb/Binarizer.h"


TEST(BinarizerTests, QuantilesTest) {
    // Example test data
    std::vector<float> heldout_avg_residual = {1.0, 2.0, 3.0, 4.0, 5.0};
    int nbits = 4;

    lintdb::Binarizer binarizer(nbits, 128);
    // Call the function to calculate bucket weights
    binarizer.calculate_quantiles(heldout_avg_residual);

    // Check the size of the bucket weights
    ASSERT_EQ(binarizer.bucket_weights.size(), 15); // In this example, nbits = 4, so 2^4 - 1 = 15
}

TEST(BinarizerTests, PackBitsTests) {
    std::vector<uint8_t> input = {1, 0, 1, 0, 1, 1, 0, 0}; // Example input data
    size_t dim = 4;
    size_t nbits = 2;

    lintdb::Binarizer binarizer(nbits, dim);

    // Test packing and unpacking
    std::vector<uint8_t> packed = binarizer.packbits(input);
    std::vector<uint8_t> unpacked = binarizer.unpackbits(packed, dim, nbits);

    // Check if the unpacked data matches the input data
    ASSERT_EQ(input, unpacked);
}

TEST(BinarizerTests, PackBitsTestLargeInput) {
    std::vector<uint8_t> input(1000, 1); // All bits set to 1
    size_t dim = 125;
    size_t nbits = 8;

    lintdb::Binarizer binarizer(nbits, dim);

    // Test packing and unpacking
    std::vector<uint8_t> packed = binarizer.packbits(input);
    std::vector<uint8_t> unpacked = binarizer.unpackbits(packed, dim, nbits);

    // Check if the unpacked data matches the input data
    ASSERT_EQ(input, unpacked);
}

TEST(BinarizerTests, BasicTest) {
    // Create a mock Binarizer instance for testing
    std::vector<float> bucket_weights = {0.1, 0.2, 0.3}; // Example bucket weights
    size_t nbits = 2; // Example number of bits
    lintdb::Binarizer binarizer(nbits, 0); // dim isn't necessary here.
    binarizer.bucket_weights = bucket_weights;

    // Generate the decompression lookup table
    std::vector<uint8_t> lut = binarizer.create_decompression_lut();

    // Perform assertions on the generated lookup table
    ASSERT_EQ(lut.size(), 9); // Expected size of lookup table for 3 bucket weights and 2 bits per lookup
    // You can perform more specific checks based on your requirements
}