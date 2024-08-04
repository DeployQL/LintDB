
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "lintdb/index.h"
#include "lintdb/query/Query.h"
#include "lintdb/query/QueryNode.h"
#include "lintdb/schema/DataTypes.h"
#include <string>
#include <fstream>
#include <iostream>

#define DATABASE_PATH "data/colbert_test.db"
#define QUERY_EMBEDDING_PATH "data/query.txt"
#define EXPECTED_RESULTS_PATH "data/colbert.ranking.tsv"

using namespace std;
/**
 * This test uses 1 query from LoTTE lifestyle and 1,000 documents.
 *
 * This is a fairly relaxed test. We ensure that the top doc ids are correct, but don't
 * enforce the order or score.
 *
 * We can notice scores change slightly between any given indexing run.
 */
TEST(ColBertTests, ScoresCorrectly) {
    auto index = lintdb::IndexIVF(DATABASE_PATH);

    // read query embeddings
    std::ifstream queryFile;
    queryFile.open(QUERY_EMBEDDING_PATH);
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

    lintdb::FieldValue fv("colbert", embeddings, 32);
    std::unique_ptr<lintdb::VectorQueryNode> root = std::make_unique<lintdb::VectorQueryNode>(fv);
    lintdb::Query query(std::move(root));

    std::vector<lintdb::SearchResult> results = index.search(0, query, 4, searchOpts);

    // print result ids and score
    for (auto& result : results) {
        std::cout << result.id << " " << result.score << std::endl;
    }

    ifstream dataFile;
    dataFile.open(EXPECTED_RESULTS_PATH);

    // read each line.
    std::unordered_set<int> doc_ids;
    int count = 0;
    while(!dataFile.eof() && count < 4) {
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
        doc_ids.insert(doc_id);
        count++;
    }

    // check if the top 10 doc ids are in the expected results.
    for (auto& result : results) {
        ASSERT_TRUE(doc_ids.find(result.id) != doc_ids.end()) << "Doc id " << result.id << " not found in expected results";
    }
}
