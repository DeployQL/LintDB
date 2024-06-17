#include <gtest/gtest.h>
#include <vector>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/index.h"

#include <memory>

TEST(EncodedDocumentTest, SerializeInvertedData) {
    std::vector<code_t> codes = {1, 2, 3};
    std::vector<residual_t> residuals = {1, 1, 1};

    lintdb::EncodedDocument doc(codes, residuals, 3, 555, 1, {});

    ASSERT_EQ(doc.code_size, 1);

    // Call the method
    auto results = doc.serialize_inverted_data();

    // Verify the results
    ASSERT_EQ(results.size(), doc.codes.size());

    for (size_t i = 0; i < doc.codes.size(); ++i) {
        EXPECT_EQ(results[i].key, doc.codes[i]);
        EXPECT_EQ(results[i].token_id, i);
        EXPECT_EQ(results[i].value.size(), 1);
    }
}