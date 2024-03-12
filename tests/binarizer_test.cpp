
#include <gtest/gtest.h>
#include <vector>
#include "lintdb/Binarizer.h"


TEST(CalculateQuantilesTest, ExampleTest) {
    // Example test data
    std::vector<float> heldout_avg_residual = {1.0, 2.0, 3.0, 4.0, 5.0};
    int nbits = 4;

    lintdb::Binarizer binarizer(nbits);
    // Call the function to calculate bucket weights
    binarizer.calculate_quantiles(heldout_avg_residual);

    // Check the size of the bucket weights
    ASSERT_EQ(binarizer.bucket_weights.size(), 15); // In this example, nbits = 4, so 2^4 - 1 = 15
}