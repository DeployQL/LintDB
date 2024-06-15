
#include <gtest/gtest.h>
#include "lintdb/index.h"
#include "lintdb/EmbeddingBlock.h"
#include <vector>
#include <map>
#include "lintdb/retrievers/XTRRetriever.h"
#include "mocks.h"

TEST(XTRTest, InitializesCorrectly) {
    auto index = std::make_shared<MockInvertedList>();
    auto forward = std::make_shared<MockForwardIndex>();
    auto encoder = std::make_shared<MockEncoder>();
    auto product_quantizer = std::make_shared<MockProductEncoder>(128, 1, 2);

    auto retriever = lintdb::XTRRetriever(index, forward, encoder, product_quantizer);
}

TEST(XTRTest, GetDocumentScores) {
    size_t n = 3; // Number of tokens per document

    std::vector<lintdb::ScoredPartialDocumentCodes> all_doc_codes = {
            {1, 0, 0, 0.1f},
            {1, 0, 1, 0.2f},
            {1, 0, 2, 0.3f},
            {2, 1, 0, 0.3f},
            {2, 1, 1, 0.1f},
            {2, 1, 2, 0.2f},
            {1, 2, 0, 0.4f},
            {1, 2, 1, 0.5f},
            {1, 2, 2, 0.1f}
    };

    std::map<idx_t, std::vector<float>> document_scores;
    vector<float> lowest_query_scores(n, numeric_limits<float>::max());

    lintdb::XTRRetriever::get_document_scores(n, all_doc_codes, document_scores, lowest_query_scores);

    // Check document_scores
    ASSERT_EQ(document_scores[1][0], 0.4f);
    ASSERT_EQ(document_scores[1][1], 0.5f);
    ASSERT_EQ(document_scores[1][2], 0.3f);

    ASSERT_EQ(document_scores[2][0], 0.3f);
    ASSERT_EQ(document_scores[2][1], 0.1f);
    ASSERT_EQ(document_scores[2][2], 0.2f);

    // Check lowest_query_scores
    ASSERT_EQ(lowest_query_scores[0], 0.1f);
    ASSERT_EQ(lowest_query_scores[1], 0.1f);
    ASSERT_EQ(lowest_query_scores[2], 0.1f);
}


TEST(XTRRetrieverTest, ImputeMissingScores) {

    size_t n = 3; // Number of tokens per document

    map<idx_t, vector<float>> document_scores = {
            {1, {0.4f, 0.5f, numeric_limits<float>::lowest()}},
            {2, {numeric_limits<float>::lowest(), 0.1f, 0.2f}},
            {3, {numeric_limits<float>::lowest(), numeric_limits<float>::lowest(), numeric_limits<float>::lowest()}}
    };

    vector<float> lowest_query_scores = {0.1f, 0.2f, 0.3f};

    lintdb::XTRRetriever::impute_missing_scores(n, document_scores, lowest_query_scores);

    // Check document_scores after imputation
    ASSERT_EQ(document_scores[1][2], 0.3f);
    ASSERT_EQ(document_scores[2][0], 0.1f);
    ASSERT_EQ(document_scores[3][0], 0.1f);
    ASSERT_EQ(document_scores[3][1], 0.2f);
    ASSERT_EQ(document_scores[3][2], 0.3f);
}

TEST(XTRRetrieverTest, StoreTokenEmbeddings) {

}