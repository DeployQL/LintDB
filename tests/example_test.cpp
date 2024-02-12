#include <gtest/gtest.h>
#include "lintdb/example.h"


// Demonstrate some basic assertions.
TEST(ExampleTest, InitializesCorrectly) {
  // std::vector<size_t> nbits = {2};
  // lintdb::ResidualQuantizer quantizer(10, nbits);
  EXPECT_EQ(lintdb::example(), true);
  // EXPECT_EQ(quantizer.code_size, 2);
  // EXPECT_EQ(quantizer.is_trained(), false);
}