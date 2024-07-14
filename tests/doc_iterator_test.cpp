#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include "lintdb/query/DocIterator.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/schema/DocEncoder.h"
#include "lintdb/schema/DocProcessor.h"

using namespace lintdb;

class VectorIterator: public Iterator {
public:
    VectorIterator(std::vector<idx_t> its): its(its) {}

    bool is_valid() override {
        return pos < its.size();
    }

    void next() override {
        pos++;
    }

    InvertedIndexKey get_key() const override {
        SupportedTypes v = 0;
        return InvertedIndexKey(0, uint8_t(0), DataType::INTEGER, v, its[pos]);

    }

    std::string get_value() const override {
        FieldValue fv;
        fv.data_type = DataType::INTEGER;
        fv.num_tensors = 0;
        fv.value = 111111;
        ProcessedData pd = ProcessedData{0, 1, {2}, its[pos], fv};
        std::vector<PostingData> encoded_value = DocEncoder::encode_inverted_data(pd, 0);
        return encoded_value[0].value;
    }

private:
    size_t pos = 0;
    std::vector<idx_t> its;
};

class MockIterator : public Iterator {
public:
    MOCK_METHOD(bool, is_valid, (), (override));
    MOCK_METHOD(void, next, (), (override));
    MOCK_METHOD(InvertedIndexKey, get_key, (), (const, override));
    MOCK_METHOD(std::string, get_value, (), (const, override));
};

class ANNIteratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // You can add setup code here
    }
};

class AndIteratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // You can add setup code here
    }
};

TEST(DocIteratorTest, AdvanceAndHasNext) {
    auto mock_it = std::make_unique<MockIterator>();
    EXPECT_CALL(*mock_it, is_valid()).WillOnce(testing::Return(true)).WillOnce(testing::Return(false));
    EXPECT_CALL(*mock_it, next()).Times(1);

    TermIterator di(std::move(mock_it));
    EXPECT_TRUE(di.is_valid());
    di.advance();
    EXPECT_FALSE(di.is_valid());
}


TEST_F(ANNIteratorTest, CorrectAggregation) {
    std::unique_ptr<VectorIterator> it1 = std::make_unique<VectorIterator>(std::vector<idx_t>({3, 4, 5}));
    std::unique_ptr<VectorIterator> it2 = std::make_unique<VectorIterator>(std::vector<idx_t>({1, 2, 6}));

    auto dit1 = std::make_unique<TermIterator>(std::move(it1));
    auto dit2 = std::make_unique<TermIterator>(std::move(it2));
    std::vector<std::unique_ptr<DocIterator>> iterators;
    iterators.push_back(std::move(dit1));
    iterators.push_back(std::move(dit2));

    ANNIterator ann(std::move(iterators));
    EXPECT_EQ(ann.doc_id(), 1);
    EXPECT_TRUE(ann.is_valid());
    ann.advance();
    EXPECT_EQ(ann.doc_id(), 2);
    ann.advance();
    EXPECT_EQ(ann.doc_id(), 3);
    ann.advance();
    EXPECT_EQ(ann.doc_id(), 4);
    ann.advance();
    EXPECT_EQ(ann.doc_id(), 5);
    ann.advance();
    EXPECT_EQ(ann.doc_id(), 6);
    ann.advance();
    EXPECT_FALSE(ann.is_valid());
}

TEST_F(AndIteratorTest, SynchronizedAdvancement) {
    std::unique_ptr<VectorIterator> it1 = std::make_unique<VectorIterator>(std::vector<idx_t>({3, 4, 5}));
    std::unique_ptr<VectorIterator> it2 = std::make_unique<VectorIterator>(std::vector<idx_t>({2, 3, 4}));

    auto dit1 = std::make_unique<TermIterator>(std::move(it1));
    auto dit2 = std::make_unique<TermIterator>(std::move(it2));
    std::vector<std::unique_ptr<DocIterator>> iterators;
    iterators.push_back(std::move(dit1));
    iterators.push_back(std::move(dit2));

    AndIterator andIt(std::move(iterators));
    EXPECT_EQ(andIt.doc_id(), 3);
    EXPECT_TRUE(andIt.is_valid());
    andIt.advance();
    EXPECT_EQ(andIt.doc_id(), 4);
    andIt.advance();
    EXPECT_FALSE(andIt.is_valid());
}

TEST_F(AndIteratorTest, SynchronizedAdvancement2) {
    std::unique_ptr<VectorIterator> it1 = std::make_unique<VectorIterator>(std::vector<idx_t>({1, 2, 3, 4}));
    std::unique_ptr<VectorIterator> it2 = std::make_unique<VectorIterator>(std::vector<idx_t>({5, 6, 7 ,8}));

    auto dit1 = std::make_unique<TermIterator>(std::move(it1));
    auto dit2 = std::make_unique<TermIterator>(std::move(it2));
    std::vector<std::unique_ptr<DocIterator>> iterators;
    iterators.push_back(std::move(dit1));
    iterators.push_back(std::move(dit2));

    ANNIterator annIt(std::move(iterators));

    std::unique_ptr<VectorIterator> it3 = std::make_unique<VectorIterator>(std::vector<idx_t>({1, 3, 5, 7}));
    std::vector<std::unique_ptr<DocIterator>> iterators2;
    iterators2.push_back(std::make_unique<ANNIterator>(std::move(annIt)));
    iterators2.push_back(std::make_unique<TermIterator>(std::move(it3)));
    AndIterator andIt(std::move(iterators2));

    EXPECT_EQ(andIt.doc_id(), 1);
    EXPECT_TRUE(andIt.is_valid());
    andIt.advance();
    EXPECT_EQ(andIt.doc_id(), 3);
    andIt.advance();
    EXPECT_EQ(andIt.doc_id(), 5);
    andIt.advance();
    EXPECT_EQ(andIt.doc_id(), 7);
    andIt.advance();
    EXPECT_FALSE(andIt.is_valid());
}