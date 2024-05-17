#include <gtest/gtest.h>
// TODO(mbarta): we introspect the invlists during tests. We can fix this with better abstractions.
#define private public
#include "lintdb/index.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/invlists/RocksdbList.h"
#include <faiss/utils/random.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <gsl/span>
#include "lintdb/util.h"
#include <unordered_set>
#include <map>

using ::testing::TestWithParam;
using ::testing::Values;

class IndexTest : public TestWithParam<lintdb::IndexEncoding> {
 public:
  ~IndexTest() override {}
  void SetUp() override { type = GetParam(); }
  void TearDown() override {
  }

 protected:
  lintdb::IndexEncoding type;
};

// Demonstrate some basic assertions.
TEST_P(IndexTest, InitializesCorrectly) {
    size_t nbits = 4; // the number of bits to encode residual codes into.
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("XXXXXX.db");

    lintdb::IndexIVF index(temp_db.string(), 5, 128, 2, 4, 16, type);

    EXPECT_EQ(index.config.nlist, 5);
}

TEST_P(IndexTest, TrainsCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 1;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("XXXXXX.db");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));

    faiss::rand_smooth_vectors(num_docs * num_tokens, dim, buf.data(), 1234);

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf);
    EXPECT_EQ(index.config.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, fake_doc.data(), 1234);
    // this doc is row-major on disk, and we can read memory as (num_tokensxdim)
    lintdb::EmbeddingBlock block(fake_doc.data(), num_tokens, dim);

    lintdb::RawPassage<lintdb::EmbeddingBlock> doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::RawPassage<lintdb::EmbeddingBlock>> docs = { doc };
    index.add(lintdb::kDefaultTenant, docs);

    // without knowing what ivf list we assigned the doc to, make sure one document is indexed.
    // this amounts to a full scan.
    for(idx_t i=0; i<kclusters; i++) {
        
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
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("XXXXXX.db");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));

    faiss::rand_smooth_vectors(num_docs * num_tokens, dim, buf.data(), 1234);

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf);
    EXPECT_EQ(index.config.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens);

    faiss::rand_smooth_vectors(num_tokens, dim, fake_doc.data(), 1234);
    // this doc is row-major on disk, and we can read memory as (num_tokensxdim)
    lintdb::EmbeddingBlock block(fake_doc.data(), num_tokens, dim);

    lintdb::RawPassage<lintdb::EmbeddingBlock> doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::RawPassage<lintdb::EmbeddingBlock>> docs = { doc };
    index.add(lintdb::kDefaultTenant, docs);

    // without knowing what ivf list we assigned the doc to, make sure one document is indexed.
    // this amounts to a full scan.
    for(idx_t i=0; i<kclusters; i++) {
        lintdb::Key start{0, i, 0, true};
        lintdb::Key end{0, i, std::numeric_limits<idx_t>::max(), false};
        std::string start_string = start.serialize();
        std::string end_string = end.serialize();
        lintdb::ReadOnlyRocksDBInvertedList casted = static_cast<lintdb::ReadOnlyRocksDBInvertedList&>(*index.index_);
        std::unique_ptr<lintdb::Iterator> it = casted.get_iterator(0, i);
        for(; it->has_next(); it->next()) {
            lintdb::Key key = it->get_key();
            auto id = key.id;
            EXPECT_EQ(id, idx_t(1));
        }
    }

}


// EmbeddingBlocks store data in column major format, so contiguous memory is
// a column of data. we should expect the first column to be the first elements generated.
// TEST(IndexTest, EmbeddingBlocksAreRowMajor) {
//     size_t dim = 128;
//     size_t num_docs = 100;
//     size_t num_tokens = 100;

//     size_t num_bits = 2;

//     std::vector<float> buf(dim * num_tokens);

//     faiss::rand_smooth_vectors(num_tokens, dim, buf.data(), 1234);
//     // let's reuse the buffer to grab what we'd consider a a block of embeddings.
//     lintdb::EmbeddingBlock block{buf.data(), num_tokens, dim};
//     //check that the first row is correct in the block.
//     for (size_t i = 0; i < num_tokens; i++) {
//         auto scal = block.embeddings[i];
//         EXPECT_EQ(scal, buf[i]);
//     }
// }

TEST_P(IndexTest, SearchCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 2;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("XXXXXX.db");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }
    // normalize before training. ColBERT returns normalized embeddings.
    lintdb::normalize_vector(buf.data(), num_docs * num_tokens, dim);

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf);

    EXPECT_EQ(index.config.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens, 3);
    lintdb::normalize_vector(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingBlock block{fake_doc.data(), num_tokens, dim};

    lintdb::RawPassage<lintdb::EmbeddingBlock> doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::RawPassage<lintdb::EmbeddingBlock>> docs = { doc };
    index.add(lintdb::kDefaultTenant, docs);

    auto opts = lintdb::SearchOptions();
    opts.centroid_score_threshold = 0;
    opts.k_top_centroids = 250;
    auto results = index.search(lintdb::kDefaultTenant, block, 64, 5, opts);

    EXPECT_GT(results.size(), 0);

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
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("XXXXXX.db");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }

    lintdb::IndexIVF* index = new lintdb::IndexIVF(temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index->train(num_docs * num_tokens, buf);

    delete index;

    auto loaded_index = lintdb::IndexIVF(temp_db.string());

    std::vector<float> query(dim * num_tokens, 1);
    loaded_index.search(lintdb::kDefaultTenant, query.data(), num_tokens, dim, 10, 5);
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
    auto temp_db = path.append("XXXXXX.db_one");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }
    // normalize before training. ColBERT returns normalized embeddings.
    lintdb::normalize_vector(buf.data(), num_docs * num_tokens, dim);

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf);

    EXPECT_EQ(index.config.nlist, 2);

    std::vector<float> fake_doc(dim * num_tokens, 3);
    lintdb::normalize_vector(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingBlock block{fake_doc.data(), num_tokens, dim};

    lintdb::RawPassage<lintdb::EmbeddingBlock> doc(fake_doc.data(), num_tokens, dim, 1);
    std::vector<lintdb::RawPassage<lintdb::EmbeddingBlock>> docs = { doc };
    index.add(lintdb::kDefaultTenant, docs);


    // create a second db.
    std::filesystem::path path_two = std::filesystem::temp_directory_path();
    auto second_db = path_two.append("XXXXXX.db_two");
    // copy the first index to create the second db.
    auto index_two = lintdb::IndexIVF(index, second_db.string());
    lintdb::RawPassage<lintdb::EmbeddingBlock> doc_two(fake_doc.data(), num_tokens, dim, 2);
    std::vector<lintdb::RawPassage<lintdb::EmbeddingBlock>> docs_two = { doc_two };
    index_two.add(lintdb::kDefaultTenant, docs_two);

    // merge the two indices.
    index.merge(second_db.string());

    index.flush();

    auto opts = lintdb::SearchOptions();
    opts.centroid_score_threshold = 0;

    std::vector<float> buf_two(dim * num_tokens, 0);
    for(size_t i=0; i< num_tokens; i++) {
        for(size_t j=0; j < dim; j++) {
            buf_two[i*dim + j] = i%11 + 1;
        }
    }
    lintdb::normalize_vector(buf_two.data(), num_tokens, dim);

    // faiss::rand_smooth_vectors(num_tokens, dim, buf_two.data(), 1234);
    lintdb::EmbeddingBlock block_two{buf_two.data(), num_tokens, dim};
    auto results = index.search(lintdb::kDefaultTenant, block_two, 2, 5, opts);

    EXPECT_EQ(results.size(), 2);
}

INSTANTIATE_TEST_SUITE_P(IndexTest, IndexTest, Values(
    lintdb::IndexEncoding::NONE, 
    lintdb::IndexEncoding::BINARIZER, 
    lintdb::IndexEncoding::PRODUCT_QUANTIZER
    ),
    [](const testing::TestParamInfo<IndexTest::ParamType>& info) {
        auto serialized = lintdb::serialize_encoding(info.param);
        return serialized;
    }
);


TEST_P(IndexTest, SearchWithMetadataCorrectly) {
    size_t dim = 128;
    // we'll generate num_docs * num_tokens random vectors for training.
    // keep in mind this needs to be larger than the number of dimensions.
    size_t num_docs = 100;
    size_t num_tokens = 100;

    size_t kclusters = 250; // number of centroids to calculate.

    size_t centroid_bits = 2;
    std::filesystem::path path = std::filesystem::temp_directory_path();
    auto temp_db = path.append("XXXXXX.db");
    // buffer for the randomly created vectors.
    // we want 128 dimension vectors for 10 tokens, for each of the 5 docs.
    std::vector<float> buf(dim * (num_docs * num_tokens));
    // fake data where every vector is either all 1s,2s...9s. 
    for(size_t i=0; i<num_docs * num_tokens; i++) {
        for(size_t j=0; j<dim; j++) {
            buf[i*dim + j] = i%11 + 1;
        }
    }
    // normalize before training. ColBERT returns normalized embeddings.
    lintdb::normalize_vector(buf.data(), num_docs * num_tokens, dim);

    lintdb::IndexIVF index(temp_db.string(), kclusters, dim, centroid_bits, 4, 16, type);

    index.train(num_docs * num_tokens, buf);

    EXPECT_EQ(index.config.nlist, 250);

    std::vector<float> fake_doc(dim * num_tokens, 3);
    lintdb::normalize_vector(fake_doc.data(), num_tokens, dim);

    lintdb::EmbeddingBlock block{fake_doc.data(), num_tokens, dim};

    lintdb::RawPassage<lintdb::EmbeddingBlock> doc(fake_doc.data(), num_tokens, dim, 1, std::map<std::string, std::string>{{"title", "test"}});
    std::vector<lintdb::RawPassage<lintdb::EmbeddingBlock>> docs = { doc };
    index.add(lintdb::kDefaultTenant, docs);

    auto opts = lintdb::SearchOptions();
    opts.centroid_score_threshold = 0;
    opts.k_top_centroids = 250;
    auto results = index.search(lintdb::kDefaultTenant, block, 64, 5, opts);

    EXPECT_GT(results.size(), 0);

    auto actual = results[0].id;
    EXPECT_EQ(actual, 1);
    EXPECT_EQ(results[0].metadata.at("title"), "test");
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}