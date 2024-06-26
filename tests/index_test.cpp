#include <gtest/gtest.h>
// TODO(mbarta): we introspect the invlists during tests. We can fix this with
// better abstractions.
#define private public
#include <faiss/utils/random.h>
#include <iostream>
#include <map>
#include <vector>
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/Passages.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/index.h"
#include "lintdb/invlists/RocksdbInvertedList.h"
#include "lintdb/util.h"
#include "util.h"

using ::testing::TestWithParam;
using ::testing::Values;

class IndexTest : public TestWithParam<lintdb::IndexEncoding> {
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
    lintdb::IndexEncoding type;
    std::filesystem::path temp_db;
    std::filesystem::path temp_db_two;
};

// Demonstrate some basic assertions.
TEST_P(IndexTest, InitializesCorrectly) {
    auto temp_db = create_temporary_directory();

    lintdb::IndexIVF index(temp_db.string(), 5, 128, 2, 4, 16, type);

    EXPECT_EQ(index.config.nlist, 5);

    EXPECT_GE(index.config.lintdb_version.major, 0);
    EXPECT_GE(index.config.lintdb_version.minor, 1);

    // we expect metadata to be enabled by default.
    EXPECT_TRUE(index.config.lintdb_version.metadata_enabled);
}

TEST_P(IndexTest, LegacytrainsCorrectly) {
    // This method tests the legacy pathway for initializing the encoder and
    // testing it. nlist and niter are no longer initialized with the encoder,
    // so we need to ensure it still works.
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 1;
    temp_db = create_temporary_directory();
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));

    faiss::rand_smooth_vectors(num_docs * num_tokens, dim, buf.data(), 1234);

    lintdb::IndexIVF index(
            temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    // legacy calling structure. nlist and niter are only passed at
    // initialization of encoder.
    index.train(num_docs * num_tokens, buf);
}

TEST_P(IndexTest, TrainsCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 1;
    temp_db = create_temporary_directory();
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));

    faiss::rand_smooth_vectors(num_docs * num_tokens, dim, buf.data(), 1234);

    lintdb::IndexIVF index(
            temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf, kclusters, 2);
    EXPECT_EQ(index.config.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, fake_doc.data(), 1234);
    // this doc is row-major on disk, and we can read memory as (num_tokensxdim)
    lintdb::EmbeddingBlock block(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingPassage doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::EmbeddingPassage> docs = {doc};
    index.add(lintdb::kDefaultTenant, docs);

    // without knowing what ivf list we assigned the doc to, make sure one
    // document is indexed. this amounts to a full scan.
    for (idx_t i = 0; i < kclusters; i++) {
        lintdb::Key start{0, i, 0, true};
        lintdb::Key end{0, i, std::numeric_limits<idx_t>::max(), false};

        auto options = rocksdb::ReadOptions();
        rocksdb::Slice end_slice(end.serialize());
        options.iterate_upper_bound = &end_slice;
        std::string start_string = start.serialize();
        auto it = std::unique_ptr<rocksdb::Iterator>(index.db->NewIterator(
                options, index.column_families[lintdb::kIndexColumnIndex]));

        rocksdb::Slice prefix(start_string);
        it->Seek(prefix);
        for (; it->Valid(); it->Next()) {
            auto k = it->key().ToString();
            auto key = lintdb::Key::from_slice(k);

            auto id = key.id;
            EXPECT_EQ(id, idx_t(1));
        }
    }
}

TEST_P(IndexTest, TrainsWithCompressionCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 2;
    temp_db = create_temporary_directory();
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));

    faiss::rand_smooth_vectors(num_docs * num_tokens, dim, buf.data(), 1234);

    lintdb::IndexIVF index(
            temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf);
    EXPECT_EQ(index.config.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, fake_doc.data(), 1234);
    // this doc is row-major on disk, and we can read memory as (num_tokensxdim)
    lintdb::EmbeddingBlock block(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingPassage doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::EmbeddingPassage> docs = {doc};
    index.add(lintdb::kDefaultTenant, docs);

    // without knowing what ivf list we assigned the doc to, make sure one
    // document is indexed. this amounts to a full scan.
    int count = 0;
    for (idx_t i = 0; i < kclusters; i++) {
        lintdb::Key start{0, i, 0, true};
        lintdb::Key end{0, i, std::numeric_limits<idx_t>::max(), false};
        std::string start_string = start.serialize();
        std::string end_string = end.serialize();
        lintdb::RocksdbInvertedList casted =
                static_cast<lintdb::RocksdbInvertedList&>(
                        reinterpret_cast<lintdb::RocksdbInvertedList&>(
                                *index.index_));
        std::unique_ptr<lintdb::Iterator> it = casted.get_iterator(0, i);
        for (; it->has_next(); it->next()) {
            lintdb::TokenKey key = it->get_key();
            auto id = key.doc_id;
            EXPECT_EQ(id, idx_t(1));
            count++;
        }
    }

    EXPECT_GE(count, 1);
}

TEST(IndexTest, RawPassagesConstruct) {
    float data = 1.0;
    auto block_passage = lintdb::EmbeddingPassage(&data, 1, 1, 1, {});

    auto text_passage = lintdb::TextPassage("test", 1, {});
}

TEST_P(IndexTest, SearchCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 2;
    temp_db = create_temporary_directory();
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s.
    for (size_t i = 0; i < num_docs * num_tokens; i++) {
        for (size_t j = 0; j < dim; j++) {
            buf[i * dim + j] = i % 11 + 1;
        }
    }
    // normalize before training. ColBERT returns normalized embeddings.
    lintdb::normalize_vector(buf.data(), num_docs * num_tokens, dim);

    lintdb::IndexIVF index(
            temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf);

    EXPECT_EQ(index.config.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens, 3);
    lintdb::normalize_vector(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingBlock block{fake_doc.data(), num_tokens, dim};

    lintdb::EmbeddingPassage doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::EmbeddingPassage> docs = {doc};
    index.add(lintdb::kDefaultTenant, docs);

    auto opts = lintdb::SearchOptions();
    opts.centroid_score_threshold = 0;
    opts.k_top_centroids = 250;
    auto results = index.search(lintdb::kDefaultTenant, block, 250, 5, opts);

    ASSERT_GT(results.size(), 0);

    auto actual = results[0].id;
    EXPECT_EQ(actual, 1);
}

TEST_P(IndexTest, LoadsCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 1;
    temp_db = create_temporary_directory();
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s.
    for (size_t i = 0; i < num_docs * num_tokens; i++) {
        for (size_t j = 0; j < dim; j++) {
            buf[i * dim + j] = i % 11 + 1;
        }
    }

    lintdb::IndexIVF* index = new lintdb::IndexIVF(
            temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);
    std::cout << temp_db.string() << std::endl;
    index->train(num_docs * num_tokens, buf);

    auto loaded_index = lintdb::IndexIVF(temp_db.string(), true);

    std::vector<float> query(dim * num_tokens, 1);
    loaded_index.search(
            lintdb::kDefaultTenant, query.data(), num_tokens, dim, 10, 5);
}

TEST_P(IndexTest, MergeCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 2; // number of centroids to calculate.

    size_t centroid_bits = 2;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    temp_db = create_temporary_directory();
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s.
    for (size_t i = 0; i < num_docs * num_tokens; i++) {
        for (size_t j = 0; j < dim; j++) {
            buf[i * dim + j] = i % 11 + 1;
        }
    }
    // normalize before training. ColBERT returns normalized embeddings.
    lintdb::normalize_vector(buf.data(), num_docs * num_tokens, dim);

    lintdb::IndexIVF index(
            temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf, kclusters, 2);

    EXPECT_EQ(index.config.nlist, 2);

    std::vector<float> fake_doc(dim * num_tokens, 3);
    lintdb::normalize_vector(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingBlock block{fake_doc.data(), num_tokens, dim};

    lintdb::EmbeddingPassage doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::EmbeddingPassage> docs = {doc};
    index.add(lintdb::kDefaultTenant, docs);

    // create a second db.
    std::filesystem::path path_two = std::filesystem::temp_directory_path();
    temp_db_two = create_temporary_directory();
    // copy the first index to create the second db.
    auto index_two = lintdb::IndexIVF(index, temp_db_two.string());
    lintdb::EmbeddingPassage doc_two(fake_doc.data(), num_tokens, dim, 2);
    std::vector<lintdb::EmbeddingPassage> docs_two = {doc_two};
    index_two.add(lintdb::kDefaultTenant, docs_two);

    // merge the two indices.
    index.merge(temp_db_two.string());
    std::cout << "merged indices" << std::endl;

    index.flush();

    auto opts = lintdb::SearchOptions();
    opts.centroid_score_threshold = 0;
    opts.nearest_tokens_to_fetch = 1000;

    std::vector<float> buf_two(dim * num_tokens, 0);
    for (size_t i = 0; i < num_tokens; i++) {
        for (size_t j = 0; j < dim; j++) {
            buf_two[i * dim + j] = i % 11 + 1;
        }
    }
    lintdb::normalize_vector(buf_two.data(), num_tokens, dim);

    // faiss::rand_smooth_vectors(num_tokens, dim, buf_two.data(), 1234);
    lintdb::EmbeddingBlock block_two{buf_two.data(), num_tokens, dim};
    auto results =
            index.search(lintdb::kDefaultTenant, block_two, 200, 5, opts);
    EXPECT_EQ(results.size(), 2);
}

TEST_P(IndexTest, SearchWithMetadataCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 2;
    temp_db = create_temporary_directory();
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s.
    for (size_t i = 0; i < num_docs * num_tokens; i++) {
        for (size_t j = 0; j < dim; j++) {
            buf[i * dim + j] = i % 11 + 1;
        }
    }
    // normalize before training. ColBERT returns normalized embeddings.
    lintdb::normalize_vector(buf.data(), num_docs * num_tokens, dim);

    lintdb::IndexIVF index(
            temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf, kclusters, 2);

    EXPECT_EQ(index.config.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens, 3);
    lintdb::normalize_vector(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingBlock block{fake_doc.data(), num_tokens, dim};

    lintdb::EmbeddingPassage doc(
            fake_doc.data(),
            num_tokens,
            dim,
            1,
            std::map<std::string, std::string>{{"title", "test"}});
    std::vector<lintdb::EmbeddingPassage> docs = {doc};
    index.add(lintdb::kDefaultTenant, docs);

    auto opts = lintdb::SearchOptions();
    opts.centroid_score_threshold = 0;
    opts.k_top_centroids = 250;
    auto results = index.search(lintdb::kDefaultTenant, block, 250, 5, opts);

    ASSERT_GT(results.size(), 0);

    auto actual = results[0].id;
    EXPECT_EQ(actual, 1);
    EXPECT_EQ(results[0].metadata.at("title"), "test");
}

INSTANTIATE_TEST_SUITE_P(
        IndexTest,
        IndexTest,
        Values(lintdb::IndexEncoding::NONE,
               lintdb::IndexEncoding::BINARIZER,
               lintdb::IndexEncoding::PRODUCT_QUANTIZER,
               lintdb::IndexEncoding::XTR),
        [](const testing::TestParamInfo<IndexTest::ParamType>& info) {
            auto serialized = lintdb::serialize_encoding(info.param);
            return serialized;
        });

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}