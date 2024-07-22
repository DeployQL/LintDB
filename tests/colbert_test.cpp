
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/index.h"
#include "lintdb/scoring/plaid.h"
#include "lintdb/util.h"
#include "lintdb/Collection.h"
#include <string>
#include <fstream>
#include <iostream>

/**
 * This test uses 1 query from LoTTE lifestyle and 1,000 documents.
 *
 * We want to ensure we're within a tolerance of ColBERT scores. We could differ
 * because of GPU/CPU usage using float16 vs float32.
 */
TEST(ColBertTests, ScoresCorrectly) {
    auto index = lintdb::IndexIVF("data/testdb");

    // read query embeddings
    ifstream queryFile;
    queryFile.open("data/query.txt");
    std::string line;
    std::vector<float> embeddings;

    while(std::getline(queryFile, line)) {
        std::stringstream buf(line);
        std::string tmp;
        while(getline(buf, tmp, ' ')) {
            float f = std::stof(tmp);
            embeddings.push_back(f);
        }
    }
    std::cout << embeddings[embeddings.size()-1] << std::endl;
    // we save a padded query, which should be 32 tokens long.
    ASSERT_EQ(embeddings.size(), 32 * 128);

    lintdb::SearchOptions searchOpts;
    searchOpts.k_top_centroids = 32;

    std::vector<lintdb::SearchResult> results = index.search(0, embeddings.data(), 32, 128, 100, searchOpts);

    ifstream dataFile;
    dataFile.open("data/colbert.ranking.tsv");

    // read each line.
    int count = 0;
    while(!dataFile.eof() && count < 10) {
        std::string str;
        std::getline( dataFile, str);
        std::stringstream buffer(str);
        std::string tmp;

        // read each column
        int doc_id;
        float doc_score;
        int ranking = 0;

        int i = 0;
        while( getline( buffer, tmp, '\t') ) {
            if (i == 1) {
                // doc id
                doc_id = std::stoi(tmp);
            }
            if (i==2) {
                ranking = std::stoi(tmp);
            }
            if (i==3) {
                doc_score = std::stof(tmp);
            }
            i++;
        }
        // we have results to compare to. rankings are 1-based.
        ASSERT_TRUE(results.size() >= ranking-1);

        // compare results. each doc and score should be equal.
        ASSERT_EQ(doc_id, results[ranking-1].id);
        ASSERT_NEAR(doc_score, results[ranking-1].score, 0.1);
        count++;
    }
}
