#pragma once

#include <vector>
#include <memory>
#include "lintdb/invlists/ContextIterator.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/query/DocIterator.h"
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/query/DocValue.h"

namespace lintdb {

    enum class ScoringType {
        UNKNOWN = 0,
        MAXSUM = 1, /// ColBERT's sum-of-max scoring on decoded vectors
        CENTROID_MAXSUM = 2 /// ColBERT's sum-of-max scoring using centroid vectors.
    };

    struct ScoringPhase {
        size_t num_docs;
        std::vector<std::string> context_fields;
        ScoringType type;

    };

    struct ScoringOptions {
        std::vector<ScoringPhase> phases;
    };


    /**
     * Scorer is an interface for scoring documents.
     *
     * Scorers will iterate over a DocIterator and score each document.
     * The caller of Scorer.score() will be responsible for keeping the scores in order.
     *
     * Additionally, different scorers can retrieve different context from fast fields.
     * For example, ColBERT will use a context field to retrieve all document codes
     * during scoring.
     */
class Scorer {
public:
    virtual ~Scorer() = default;
    virtual float score(idx_t doc_id, std::vector<DocValue>& dvs) const = 0;
};

class ColBERTScorer: public Scorer {
public:
    ColBERTScorer(std::shared_ptr<InvertedList> index, uint64_t tenant, uint8_t field_id, size_t code_size);
    float score(idx_t doc_id, std::vector<DocValue>& fvs) const override;
    ~ColBERTScorer() override = default;
private:
    uint8_t field_id;
    uint64_t tenant;
    size_t code_size;
    std::unique_ptr<ContextIterator> it;
};

//class XTRScorer: public Scorer {
//    double score(idx_t doc_id, std::vector<FieldValue>& fvs) const override;
//};

}
