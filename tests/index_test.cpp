#include <gtest/gtest.h>
// TODO(mbarta): we introspect the invlists during tests. We can fix this with
// better abstractions.
#define private public
#include "lintdb/SearchOptions.h"
#include "lintdb/index.h"
#include "lintdb/util.h"
#include "util.h"
#include "lintdb/schema/Schema.h"
#include "lintdb/schema/Document.h"
#include "lintdb/query/Query.h"
#include "lintdb/query/QueryNode.h"

using ::testing::TestWithParam;
using ::testing::Values;

lintdb::Schema create_colbert_schema(lintdb::QuantizerType type = lintdb::QuantizerType::NONE, size_t num_centroids = 10, std::vector<lintdb::DataType> filter_types = {}){
    lintdb::Schema schema;

    lintdb::Field colbert;
    colbert.name = "colbert";
    colbert.data_type = lintdb::DataType::TENSOR;
    colbert.field_types = {lintdb::FieldType::Colbert};
    lintdb::FieldParameters fp;
    fp.dimensions = 128;
    fp.num_centroids = num_centroids;
    fp.num_iterations = 2;
    fp.quantization = type;
    fp.nbits = 1;
    if ( type == lintdb::QuantizerType::PRODUCT_ENCODER) {
        fp.num_subquantizers = 16;
    }
    colbert.parameters = fp;

    schema.add_field(colbert);

    if(!filter_types.empty()) {
        for(size_t i=0 ; i < filter_types.size(); i++) {
            lintdb::Field filter;
            filter.name = "filter" + std::to_string(i);
            filter.data_type = filter_types[i];
            filter.field_types = {lintdb::FieldType::Indexed};
            lintdb::FieldParameters fpp;
            filter.parameters = fpp;

            schema.add_field(filter);
        }
    }

    return schema;
}

lintdb::Document create_document(size_t num_tokens, size_t dim, const std::vector<std::string> filters = {}, const std::vector<lintdb::FieldValue> filter_values = {}){
    std::vector<float> vector;
    for (size_t j = 0; j < num_tokens; j++) {
        std::vector<float> data(dim, j);
        vector.insert(vector.end(), data.begin(), data.end());
    }
    lintdb::FieldValue fv("colbert", vector, num_tokens);
    std::vector<lintdb::FieldValue> fields = {fv};
    for(int i = 0; i < filters.size(); i++) {
        fields.push_back(filter_values[i]);
    }
    lintdb::Document doc(0, fields );

    assert(filters.size() == filter_values.size());



    return doc;
}

std::vector<lintdb::Document> create_colbert_documents(size_t num_docs, size_t num_tokens, size_t dim, std::vector<lintdb::DataType> filter_types = {}){
    std::vector<lintdb::Document> docs;
    for (idx_t i = 0; i < num_docs; i++) {
        std::vector<float> vector;
        for (size_t j = 0; j < num_tokens; j++) {
            std::vector<float> data(dim, j);
            vector.insert(vector.end(), data.begin(), data.end());
        }
        lintdb::FieldValue fv("colbert", vector, num_tokens);

        std::vector<lintdb::FieldValue> filter_values({fv});
        if (!filter_types.empty()) {
            for(int i = 0; i < filter_types.size(); i++) {
                if (filter_types[i] == lintdb::DataType::TEXT) {
                    lintdb::FieldValue text_value("filter" + std::to_string(i), "test");
                    filter_values.push_back(text_value);
                } else if (filter_types[i] == lintdb::DataType::INTEGER) {
                    lintdb::FieldValue int_value("filter" + std::to_string(i), 1);
                    filter_values.push_back(int_value);
                }
            }
        }

        lintdb::Document doc(i, filter_values);

        docs.push_back(doc);
    }
    return docs;
}

class IndexTest : public TestWithParam<lintdb::QuantizerType> {
   public:
    ~IndexTest() override {}
    void SetUp() override {
        type = GetParam();
    }

    void TearDown() override {
        std::filesystem::remove_all(temp_db);
        if (!temp_db_two.empty()) {
            std::filesystem::remove_all(temp_db_two);
        }
    }

   protected:
    std::filesystem::path temp_db;
    std::filesystem::path temp_db_two;
    lintdb::QuantizerType type;
};

// Demonstrate some basic assertions.
TEST_P(IndexTest, InitializesCorrectly) {
    auto temp_db = create_temporary_directory();

    lintdb::Configuration config;
    lintdb::Schema schema = create_colbert_schema(type);
    lintdb::IndexIVF index(
            temp_db.string(), schema, config);

    EXPECT_GE(index.config.lintdb_version.major, 0);
    EXPECT_GE(index.config.lintdb_version.minor, 1);

    // we expect metadata to be enabled by default.
    EXPECT_TRUE(index.config.lintdb_version.metadata_enabled);
}

TEST_P(IndexTest, TrainsCorrectly) {
    temp_db = create_temporary_directory();

    lintdb::Configuration config;
    lintdb::Schema schema = create_colbert_schema(type);
    lintdb::IndexIVF index(
            temp_db.string(), schema, config);

    auto docs = create_colbert_documents(20, 10, 128);

    index.train(docs);
    EXPECT_EQ(index.coarse_quantizer_map.size(), 1);
    // we only expect a quantizer when the quantizer type isn't NONE
    EXPECT_EQ(index.quantizer_map.size(),  1);
}

TEST_P(IndexTest, SearchCorrectly) {
    temp_db = create_temporary_directory();

    lintdb::Configuration config;
    lintdb::Schema schema = create_colbert_schema(type, 10);
    lintdb::IndexIVF index(
            temp_db.string(), schema, config);

    auto docs = create_colbert_documents(10, 10, 128);
    index.train(docs);

    index.add(1, docs);

    lintdb::FieldValue fv("colbert", std::vector<float>(1280, 1), 10);
    std::unique_ptr<lintdb::VectorQueryNode> root = std::make_unique<lintdb::VectorQueryNode>(fv);
    lintdb::Query query(std::move(root));

    lintdb::SearchOptions opt;
    opt.n_probe = 100;
    opt.k_top_centroids = 10;

    auto results = index.search(1, query, 10, opt);

    // create a set of range 0...9
    std::set<idx_t> expected;
    for (idx_t i = 0; i < 10; i++) {
        expected.insert(i);
    }

    EXPECT_EQ(results.size(), 10);

    // check that the results are in the expected set.
    for (auto& result : results) {
        EXPECT_TRUE(expected.find(result.id) != expected.end()) << "id: " << result.id;
        // remove the result from the set.
        expected.erase(result.id);
    }

}

TEST_P(IndexTest, SearchCorrectlyWithFilter) {
    temp_db = create_temporary_directory();

    lintdb::Configuration config;
    lintdb::Schema schema = create_colbert_schema(type, 10, {lintdb::DataType::TEXT, lintdb::DataType::INTEGER});
    lintdb::IndexIVF index(
            temp_db.string(), schema, config);

    auto docs = create_colbert_documents(10, 10, 128, {lintdb::DataType::TEXT, lintdb::DataType::INTEGER});
    index.train(docs);

    index.add(1, docs);
    lintdb::FieldValue fv("colbert", std::vector<float>(1280, 1), 10);
    std::unique_ptr<lintdb::VectorQueryNode> vector_node = std::make_unique<lintdb::VectorQueryNode>(fv);
    lintdb::FieldValue text_value("filter0", "test");
    std::unique_ptr<lintdb::QueryNode> text_node = std::make_unique<lintdb::TermQueryNode>(text_value);
    lintdb::FieldValue int_value("filter1", 1);
    std::unique_ptr<lintdb::QueryNode> int_node = std::make_unique<lintdb::TermQueryNode>(int_value);

    std::vector<std::unique_ptr<lintdb::QueryNode>> children;
    children.push_back(std::move(vector_node));
    children.push_back(std::move(text_node));
    children.push_back(std::move(int_node));
    std::unique_ptr<lintdb::QueryNode> root = std::make_unique<lintdb::AndQueryNode>(std::move(children));

    lintdb::Query query(std::move(root));

    lintdb::SearchOptions opt;
    opt.n_probe = 100;
    opt.k_top_centroids = 10;

    auto results = index.search(1, query, 50, opt);
    // create a set of range 0...9
    std::set<idx_t> expected;
    for (idx_t i = 0; i < 10; i++) {
        expected.insert(i);
    }

    EXPECT_EQ(results.size(), 10);

    // check that the results are in the expected set.
    for (auto& result : results) {
        EXPECT_TRUE(expected.find(result.id) != expected.end()) << "id: " << result.id;
        // remove the result from the set.
        expected.erase(result.id);
    }

    // add docs with only a text filter.
    auto text_docs = create_colbert_documents(10, 10, 128, {lintdb::DataType::TEXT});
    // fudge the doc id to be different.
    for (auto& doc : text_docs) {
        doc.id += 10;
    }

    index.add(1, text_docs);

    // when we search with the integer filter, we should get back the same ten results.
    lintdb::FieldValue int_query_value("filter1", 1);
    std::unique_ptr<lintdb::QueryNode> int_query_node = std::make_unique<lintdb::TermQueryNode>(int_query_value);
    std::unique_ptr<lintdb::VectorQueryNode> int_vector_node = std::make_unique<lintdb::VectorQueryNode>(fv);
    std::vector<std::unique_ptr<lintdb::QueryNode>> int_children;
    int_children.push_back(std::move(int_vector_node));
    int_children.push_back(std::move(int_query_node));
    std::unique_ptr<lintdb::QueryNode> int_root = std::make_unique<lintdb::AndQueryNode>(std::move(int_children));
    lintdb::Query int_query(std::move(int_root));

    auto int_results = index.search(1, int_query, 50, opt);
    EXPECT_EQ(int_results.size(), 10);

    std::set<idx_t> int_expected;
    for (idx_t i = 0; i < 10; i++) {
        int_expected.insert(i);
    }

    // check that the results are in the expected set.
    for (auto& result : int_results) {
        EXPECT_TRUE(int_expected.find(result.id) != int_expected.end()) << "id: " << result.id;
        // remove the result from the set.
        int_expected.erase(result.id);
    }

    // when we search with the text filter, we should get back 20 results.
    lintdb::FieldValue text_query_value("filter0", "test");
    std::unique_ptr<lintdb::QueryNode> text_query_node = std::make_unique<lintdb::TermQueryNode>(text_query_value);
    std::vector<std::unique_ptr<lintdb::QueryNode>> text_children;

    std::unique_ptr<lintdb::VectorQueryNode> text_vector_node = std::make_unique<lintdb::VectorQueryNode>( fv);
    text_children.push_back(std::move(text_vector_node));
    text_children.push_back(std::move(text_query_node));
    std::unique_ptr<lintdb::QueryNode> text_root = std::make_unique<lintdb::AndQueryNode>(std::move(text_children));
    lintdb::Query text_query(std::move(text_root));

    auto text_results = index.search(1, text_query, 50, opt);
    EXPECT_EQ(text_results.size(), 20) << "expected the original 10 and the 10 with only text filter";

    std::set<idx_t> text_expected;
    for (idx_t i = 0; i < 20; i++) {
        text_expected.insert(i);
    }

    // check that the results are in the expected set.
    for (auto& result : text_results) {
        EXPECT_TRUE(text_expected.find(result.id) != text_expected.end()) << "id: " << result.id;
        // remove the result from the set.
        text_expected.erase(result.id);
    }

    // now let's search for a document with a text filter that doesn't exist.
    lintdb::FieldValue text_value_two("filter0", "test2");
    std::unique_ptr<lintdb::QueryNode> text_node_two = std::make_unique<lintdb::TermQueryNode>(text_value_two);

    auto text_children_two = std::vector<std::unique_ptr<lintdb::QueryNode>>();
    std::unique_ptr<lintdb::VectorQueryNode> text_two_vector_node = std::make_unique<lintdb::VectorQueryNode>( fv);
    text_children_two.push_back(std::move(text_two_vector_node));
    text_children_two.push_back(std::move(text_node_two));
    std::unique_ptr<lintdb::QueryNode> text_root_two = std::make_unique<lintdb::AndQueryNode>(std::move(text_children_two));

    lintdb::Query text_query_two(std::move(text_root_two));

    auto text_results_two = index.search(1, text_query_two, 50, opt);
    EXPECT_EQ(text_results_two.size(), 0) << "should not find any results for a non-existent text filter";
}

TEST_P(IndexTest, LoadsCorrectly) {
    temp_db = create_temporary_directory();

    lintdb::Configuration config;
    lintdb::Schema schema = create_colbert_schema(type);
    lintdb::IndexIVF index(
            temp_db.string(), schema, config);

    auto docs = create_colbert_documents(10, 10, 128);

    index.train(docs);

    std::cout << "HERE" << std::endl;
    index.add(1, docs);
    std::cout << "HERE" << std::endl;
    auto loaded_index = lintdb::IndexIVF(temp_db.string(), true);

    lintdb::FieldValue fv("colbert", std::vector<float>(1280, 1), 10);
    std::unique_ptr<lintdb::VectorQueryNode> root = std::make_unique<lintdb::VectorQueryNode>(fv);
    lintdb::Query query(std::move(root));
    std::cout << "HERE" << std::endl;
    auto one_results = index.search(
            1, query, 10);
    std::cout << "HERE" << std::endl;
    lintdb::SearchOptions opt;
    opt.n_probe = 100;
    opt.k_top_centroids = 10;
    auto results = loaded_index.search(
            1, query, 10, opt);
    std::cout << "HERE" << std::endl;
    EXPECT_EQ(results.size(), one_results.size());

    index.close();
    loaded_index.close();
}

TEST_P(IndexTest, MergeCorrectly) {
    temp_db = create_temporary_directory();

    lintdb::Configuration config;
    lintdb::Schema schema = create_colbert_schema(type);
    lintdb::IndexIVF index(
            temp_db.string(), schema, config);

    auto docs = create_colbert_documents(10, 10, 128);

    index.train(docs);

    index.add(1, {docs[0]});

    index.save();
    std::cout << "first file path: " << temp_db.string() << std::endl;
    temp_db_two = create_temporary_directory();
    std::cout << "second file path: " << temp_db_two.string() << std::endl;
//
//    // create a second db.
    std::filesystem::path path_two = std::filesystem::temp_directory_path();
    // copy the first index to create the second db.
    // this makes it simpler to add a document, since we don't need to retrain.
    lintdb::IndexIVF index_two(
            index, temp_db_two.string());

    index_two.add(1, {docs[1]});

    // merge the two indices.
    index.merge(temp_db_two.string());

    index.flush();

    auto opts = lintdb::SearchOptions();
    opts.n_probe = 100;
    opts.k_top_centroids = 10;

    lintdb::FieldValue fv("colbert", std::vector<float>(1280, 1), 10);
    std::unique_ptr<lintdb::VectorQueryNode> root = std::make_unique<lintdb::VectorQueryNode>(fv);
    lintdb::Query query(std::move(root));

    auto results = index.search(1, query, 5, opts);
    EXPECT_EQ(results.size(), 2);
}

//TEST_P(IndexTest, SearchWithMetadataCorrectly) {
//    size_t dim = 128;
//    // we'll generate num_docs * num_tokens random vectors for training.
//    // keep in mind this needs to be larger than the number of dimensions.
//    size_t num_docs = 100;
//    size_t num_tokens = 100;
//
//    size_t kclusters = 250; // number of centroids to calculate.
//
//    size_t centroid_bits = 2;
//    temp_db = create_temporary_directory();
//    // buffer for the randomly created vectors.
//    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
//    std::vector<float> buf(dim * (num_docs * num_tokens));
//    // fake data where every vector is either all 1s,2s...9s.
//    for (size_t i = 0; i < num_docs * num_tokens; i++) {
//        for (size_t j = 0; j < dim; j++) {
//            buf[i * dim + j] = i % 11 + 1;
//        }
//    }
//    // normalize before training. ColBERT returns normalized embeddings.
//    lintdb::normalize_vector(buf.data(), num_docs * num_tokens, dim);
//
//    lintdb::IndexIVF index(
//            temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);
//
//    index.train(num_docs * num_tokens, buf, kclusters, 2);
//
//
//    std::vector<float> fake_doc(dim * num_tokens, 3);
//    lintdb::normalize_vector(fake_doc.data(), num_tokens, dim);
//
//    lintdb::EmbeddingBlock block{fake_doc.data(), num_tokens, dim};
//
//    lintdb::EmbeddingPassage doc(
//            fake_doc.data(),
//            num_tokens,
//            dim,
//            1,
//            std::map<std::string, std::string>{{"title", "test"}});
//    std::vector<lintdb::EmbeddingPassage> docs = {doc};
//    index.add(lintdb::kDefaultTenant, docs);
//
//    auto opts = lintdb::SearchOptions();
//    opts.centroid_score_threshold = 0;
//    opts.k_top_centroids = 250;
//    auto results = index.search(lintdb::kDefaultTenant, block, 250, 5, opts);
//
//    ASSERT_GT(results.size(), 0);
//
//    auto actual = results[0].id;
//    EXPECT_EQ(actual, 1);
//    EXPECT_EQ(results[0].metadata.at("title"), "test");
//}

INSTANTIATE_TEST_SUITE_P(
        IndexTest,
        IndexTest,
        Values(lintdb::QuantizerType::NONE,
               lintdb::QuantizerType::BINARIZER,
               lintdb::QuantizerType::PRODUCT_ENCODER
               ),
        [](const testing::TestParamInfo<IndexTest::ParamType>& info) {
            return std::to_string(static_cast<int>(info.param));
        });

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}