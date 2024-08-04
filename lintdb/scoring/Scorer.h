#pragma once

#include <memory>
#include <vector>
#include "lintdb/invlists/ContextIterator.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/query/DocIterator.h"
#include "lintdb/query/DocValue.h"
#include "lintdb/query/QueryContext.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/scoring/plaid.h"

namespace lintdb {

struct ScoredDocument {
    float score;
    idx_t doc_id;
    std::vector<DocValue>
            values; /// ScoredDocument takes ownership of the values, because
    /// we assume we are iterating over a DocIterator and the values are only
    /// valid for the duration of the iteration.

    ScoredDocument() = default;

    ScoredDocument(float score, idx_t doc_id, std::vector<DocValue> values)
            : score(score), doc_id(doc_id), values(std::move(values)) {}

    bool operator<(const ScoredDocument& other) const {
        return score < other.score;
    }

    bool operator>(const ScoredDocument& other) const {
        return score > other.score;
    }
};

/**
 * Scorer is an interface for scoring documents.
 *
 * Scorers will iterate over a DocIterator and score each document.
 * The caller of Scorer.score() will be responsible for keeping the scores in
 * order.
 *
 * Additionally, different scorers can retrieve different context from fast
 * fields. For example, ColBERT will use a context field to retrieve all
 * document codes during scoring.
 */
class Scorer {
   public:
    virtual ~Scorer() = default;
    virtual ScoredDocument score(
            QueryContext& context,
            idx_t doc_id,
            std::vector<DocValue>& fvs) const = 0;
};

class PlaidScorer : public Scorer {
   public:
    explicit PlaidScorer(const QueryContext& context);
    ScoredDocument score(
            QueryContext& context,
            idx_t doc_id,
            std::vector<DocValue>& fvs) const override;
    ~PlaidScorer() override = default;

   private:
    std::unique_ptr<ContextIterator> colbert_it;
};

class ColBERTScorer : public Scorer {
   public:
    explicit ColBERTScorer(const QueryContext& context);
    ScoredDocument score(
            QueryContext& context,
            idx_t doc_id,
            std::vector<DocValue>& fvs) const override;
    ~ColBERTScorer() override = default;

   private:
    std::unique_ptr<ContextIterator> colbert_it;
};

// class XTRScorer: public Scorer {
//     double score(idx_t doc_id, std::vector<FieldValue>& fvs) const override;
// };

} // namespace lintdb
