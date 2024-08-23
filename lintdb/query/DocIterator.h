#pragma once

#include <memory>
#include <vector>
#include "DocValue.h"
#include "lintdb/api.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/scoring/scoring_methods.h"
#include "lintdb/scoring/ScoredDocument.h"
#include "lintdb/invlists/ContextIterator.h"
#include "lintdb/scoring/ContextCollector.h"
#include "lintdb/query/KnnNearestCentroids.h"

namespace lintdb {
/**
 * DocIterator helps us iterate over documents in the index.
 *
 * This encapsulates the logic of and/or and aligning index iterators to the
 * same document.
 */
class DocIterator {
   public:
    virtual void advance() = 0;
    virtual bool is_valid() = 0;

    virtual idx_t doc_id() const = 0;
    virtual std::vector<DocValue> fields() const = 0;
    virtual ScoredDocument score(std::vector<DocValue> fields) const = 0;

    virtual ~DocIterator() = default;
};

class TermIterator : public DocIterator {
   private:
    std::unique_ptr<Iterator> it_;

   public:
    TermIterator() = default;
    explicit TermIterator(
            std::unique_ptr<Iterator> it,
            DataType type,
            UnaryScoringMethod scoring_method,
            bool ignore_value = false);
    void advance() override;
    bool is_valid() override;

    idx_t doc_id() const override;
    std::vector<DocValue> fields() const override;
    ScoredDocument score(std::vector<DocValue> fields) const override;

   private:
    bool ignore_value;
    DataType type;
    UnaryScoringMethod scoring_method;
};

class ANNIterator : public DocIterator {
   public:
    explicit ANNIterator(std::vector<std::unique_ptr<DocIterator>> its,
                         ContextCollector context_collector,
                         std::shared_ptr<KnnNearestCentroids> knn,
                         EmbeddingScoringMethod scoring_method);
    void advance() override;
    bool is_valid() override;

    idx_t doc_id() const override;
    std::vector<DocValue> fields() const override;
    ScoredDocument score(std::vector<DocValue> fields) const override;

   private:
    std::vector<DocValue> fields_;
    std::vector<std::unique_ptr<DocIterator>> its_;
    ContextCollector context_collector;
    std::shared_ptr<KnnNearestCentroids> knn_;
    idx_t last_doc_id_;
    EmbeddingScoringMethod scoring_method;
    void heapify(size_t idx);
};

class AndIterator : public DocIterator {
   private:
    std::vector<std::unique_ptr<DocIterator>> its_;
    idx_t current_doc_id_;
    bool is_valid_;
    NaryScoringMethod scoring_method;

    void synchronize();

   public:
    AndIterator(std::vector<std::unique_ptr<DocIterator>> iterators,
                NaryScoringMethod scoring_method);
    void advance() override;
    bool is_valid() override;
    idx_t doc_id() const override;
    std::vector<DocValue> fields() const override;
    ScoredDocument score(std::vector<DocValue> fields) const override;
};

class OrIterator : public DocIterator {
   public:
    explicit OrIterator(std::vector<std::unique_ptr<DocIterator>> its,
                        NaryScoringMethod scoring_method);
    void advance() override;
    bool is_valid() override;

    idx_t doc_id() const override;
    std::vector<DocValue> fields() const override;
    ScoredDocument score(std::vector<DocValue> fields) const override;

   private:
    std::vector<DocValue> fields_;
    std::vector<std::unique_ptr<DocIterator>> its_;
    idx_t last_doc_id_;
    NaryScoringMethod scoring_method;
    void heapify(size_t idx);
};

} // namespace lintdb
