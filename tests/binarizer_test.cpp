
#include <gtest/gtest.h>
#include <vector>
#define private public
#include <cmath>
#include <random>
#include "lintdb/quantizers/Binarizer.h"
#include "lintdb/utils/endian.h"

TEST(BinarizerTests, QuantilesTest) {
    // Example test data
    std::vector<float> heldout_avg_residual = {1.0, 2.0, 3.0, 4.0, 5.0};
    int nbits = 4;

    lintdb::Binarizer binarizer(nbits, 128);
    // Call the function to calculate bucket weights
    binarizer.calculate_quantiles(heldout_avg_residual);

    // Check the size of the bucket weights
    ASSERT_EQ(
            binarizer.bucket_weights.size(),
            16); // In this example, nbits = 4, so 2^4 - 1 = 15
}

TEST(BinarizerTests, PackBitsTests) {
    // input is dim * nbits. we assume we've already binarized the data.
    std::vector<uint8_t> input = {
            1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
            0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0}; // Example input data
    size_t dim = 16;
    size_t nbits = 2;

    lintdb::Binarizer binarizer(nbits, dim);

    // Test packing and unpacking
    std::vector<uint8_t> packed = binarizer.packbits(input);
    ASSERT_EQ(
            packed.size(),
            dim / 8 * nbits); // Expected size of packed data (2 bits per value,
                              // 16 values = 16 / 8 * 2 = 4 bytes)

    std::vector<uint8_t> unpacked = binarizer.unpackbits(packed, dim, nbits);
    ASSERT_EQ(unpacked.size(), dim * nbits); // Expected size of unpacked data
    // Check if the unpacked data matches the input data
    ASSERT_EQ(input, unpacked);
}

TEST(BinarizerTests, PackBitsTestLargeInput) {
    size_t dim = 128;
    size_t nbits = 2;
    std::vector<uint8_t> input(128 * nbits, 1); // All bits set to 1

    lintdb::Binarizer binarizer(nbits, dim);

    // Test packing and unpacking
    std::vector<uint8_t> packed = binarizer.packbits(input);
    std::vector<uint8_t> unpacked = binarizer.unpackbits(packed, dim, nbits);

    // Check if the unpacked data matches the input data
    ASSERT_EQ(input, unpacked);
}

TEST(BinarizerTests, DecompressionLUT) {
    // Create a mock Binarizer instance for testing
    std::vector<float> bucket_weights = {
            0.1, 0.2, 0.3}; // Example bucket weights
    size_t nbits = 2;       // Example number of bits
    lintdb::Binarizer binarizer(
            nbits,
            16); // dim isn't necessary here, but we must pass it to init.
    binarizer.bucket_weights = bucket_weights;

    // Generate the decompression lookup table
    std::vector<uint8_t> lut = binarizer.create_decompression_lut();

    size_t keys_per_byte = 8 / nbits;
    size_t num_keys = bucket_weights.size();
    size_t num_entries = pow(num_keys, keys_per_byte);

    // Perform assertions on the generated lookup table
    ASSERT_EQ(
            lut.size(),
            num_entries *
                    keys_per_byte); // Expected size of lookup table for 3
                                    // bucket weights and 2 bits per lookup
}

TEST(BinarizerTests, ReverseBitmapNbits1) {
    // Create a mock Binarizer instance for testing
    std::vector<float> bucket_weights = {
            0.1, 0.2, 0.3}; // Example bucket weights
    size_t nbits = 1;       // Example number of bits
    lintdb::Binarizer binarizer(
            nbits,
            16); // dim isn't necessary here, but we must pass it to init.
    binarizer.bucket_weights = bucket_weights;

    // Generate the decompression lookup table
    std::vector<uint8_t> bm = binarizer.create_reverse_bitmap();

    // Perform assertions on the generated lookup table
    ASSERT_EQ(bm.size(), 256);

    for (size_t i = 0; i < 256; i++) {
        uint8_t reverse = bm[i];
        std::vector<unsigned char> expected;
        lintdb::store_bigendian(uint8_t(i), expected);
        uint8_t casted = expected[0];
        ASSERT_EQ(reverse, casted); // with nbits==1, the reverse bitmap should
                                    // be the same as the input.
    }
}

TEST(BinarizerTests, EncodingTest) {
    // we get the number of buckets by 1 << nbits. 1 << 2 == 4
    size_t nbits = 2;                       // Example number of bits
    lintdb::Binarizer binarizer(nbits, 16); // harcoding dim as 8.

    std::vector<float> input = {
            0.1,
            0.2,
            0.3,
            0.1,
            0.2,
            0.3,
            0.1,
            0.2,
            0.1,
            0.2,
            0.3,
            0.1,
            0.2,
            0.3,
            0.1,
            0.2}; // Example input data
    binarizer.train(1, input.data(), 16);

    std::vector<uint8_t> output(16 / 8 * nbits, 0); //
    binarizer.sa_encode(1, input.data(), output.data());

    std::vector<float> decoded(16, 0);
    binarizer.sa_decode(1, output.data(), decoded.data());

    ASSERT_EQ(input, decoded);
}

// Helper function to generate random residuals
std::vector<uint8_t> generateRandomResiduals(size_t n, size_t packed_dim) {
    std::vector<uint8_t> residuals(n * packed_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (size_t i = 0; i < residuals.size(); ++i) {
        residuals[i] = dis(gen);
    }

    return residuals;
}

TEST(BinarizerTests, CompareOriginalAndOptimized) {
    lintdb::Binarizer binarizer(1, 128);
    binarizer.bucket_weights = {-1.0f, 1.0f};
    binarizer.reverse_bitmap = binarizer.create_reverse_bitmap();
    binarizer.decompression_lut = binarizer.create_decompression_lut();

    const size_t n = 1000;  // Number of vectors to decode
    const size_t packed_dim = binarizer.dim / 8;  // 1 bit per value, so 8 values per byte

    auto residuals = generateRandomResiduals(n, packed_dim);

    std::vector<float> original_output(n * binarizer.dim);
    std::vector<float> optimized_output(n * binarizer.dim);

    // Call the original method
    binarizer.sa_decode_generic(n, residuals.data(), original_output.data());

    // Call the optimized method
    binarizer.sa_decode_1bit(n, residuals.data(), optimized_output.data());

    // Compare results
    for (size_t i = 0; i < n * binarizer.dim; ++i) {
        EXPECT_FLOAT_EQ(original_output[i], optimized_output[i])
                << "Mismatch at index " << i;
    }
}

// Test case: Check if all possible bit patterns are handled correctly
TEST(BinarizerTests, DISABLED_AllBitPatterns) {
    lintdb::Binarizer binarizer(1, 128);
    binarizer.bucket_weights = {-1.0f, 1.0f};
    binarizer.reverse_bitmap = binarizer.create_reverse_bitmap();
    binarizer.decompression_lut = binarizer.create_decompression_lut();

    const size_t n = 256;  // Test all possible byte values
    const size_t packed_dim = 1;  // One byte per vector

    std::vector<uint8_t> residuals(n);
    for (int i = 0; i < 256; ++i) {
        residuals[i] = static_cast<uint8_t>(i);
    }

    std::vector<float> original_output(n * 8);  // 8 values per byte
    std::vector<float> optimized_output(n * 8);

    binarizer.sa_decode_generic(n, residuals.data(), original_output.data());
    binarizer.sa_decode_1bit(n, residuals.data(), optimized_output.data());

    for (size_t i = 0; i < n * 8; ++i) {
        EXPECT_FLOAT_EQ(original_output[i], optimized_output[i])
                << "Mismatch at index " << i;
    }
}

// Test case: Edge case with all zeros
TEST(BinarizerTests, AllZeros) {
    lintdb::Binarizer binarizer(1, 128);
    binarizer.bucket_weights = {-1.0f, 1.0f};
    binarizer.reverse_bitmap = binarizer.create_reverse_bitmap();
    binarizer.decompression_lut = binarizer.create_decompression_lut();

    const size_t n = 100;
    const size_t packed_dim = binarizer.dim / 8;

    std::vector<uint8_t> residuals(n * packed_dim, 0);
    std::vector<float> original_output(n * binarizer.dim);
    std::vector<float> optimized_output(n * binarizer.dim);

    binarizer.sa_decode_generic(n, residuals.data(), original_output.data());
    binarizer.sa_decode_1bit(n, residuals.data(), optimized_output.data());

    for (size_t i = 0; i < n * binarizer.dim; ++i) {
        EXPECT_FLOAT_EQ(original_output[i], optimized_output[i])
                << "Mismatch at index " << i;
    }
}

// Test case: Edge case with all ones
TEST(BinarizerTests, AllOnes) {
    lintdb::Binarizer binarizer(1, 128);
    binarizer.bucket_weights = {-1.0f, 1.0f};
    binarizer.reverse_bitmap = binarizer.create_reverse_bitmap();
    binarizer.decompression_lut = binarizer.create_decompression_lut();

    const size_t n = 100;
    const size_t packed_dim = binarizer.dim / 8;

    std::vector<uint8_t> residuals(n * packed_dim, 255);
    std::vector<float> original_output(n * binarizer.dim);
    std::vector<float> optimized_output(n * binarizer.dim);

    binarizer.sa_decode_generic(n, residuals.data(), original_output.data());
    binarizer.sa_decode_1bit(n, residuals.data(), optimized_output.data());

    for (size_t i = 0; i < n * binarizer.dim; ++i) {
        EXPECT_FLOAT_EQ(original_output[i], optimized_output[i])
                << "Mismatch at index " << i;
    }
}
