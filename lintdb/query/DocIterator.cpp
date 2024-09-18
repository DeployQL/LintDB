#include "DocIterator.h"
#include <glog/logging.h>
#include "DocValue.h"
#include "lintdb/schema/DocEncoder.h"
#include "lintdb/scoring/ScoredDocument.h"

namespace lintdb {
TermIterator::TermIterator(
        std::unique_ptr<Iterator> it,
        DataType type,
        UnaryScoringMethod scoring_method,
        bool ignore_value)
        : it_(std::move(it)), ignore_value(ignore_value), type(type), scoring_method(scoring_method) {}

void TermIterator::advance() {
    it_->next();
}

bool TermIterator::is_valid() {
    return it_->is_valid();
}

idx_t TermIterator::doc_id() const {
    return it_->get_key().doc_id();
}
std::vector<DocValue> TermIterator::fields() const {
    std::string value = it_->get_value();
    auto field_id = it_->get_key().field();

    SupportedTypes doc_value;
    if (!ignore_value) {
        doc_value = DocEncoder::decode_supported_types(value);
    }

    DocValue result = DocValue(doc_value, field_id, type);
    result.unread_value = ignore_value;

    return {result};
}

ScoredDocument TermIterator::score(std::vector<DocValue> fields) const {
//    LOG(INFO) << "Term Scoring method";
    score_t score = lintdb::score(this->scoring_method, fields);

    return ScoredDocument(score, doc_id(), fields);
}

ANNIterator::ANNIterator(std::vector<std::unique_ptr<DocIterator>> its,
                         ContextCollector context_collector,
                         std::shared_ptr<KnnNearestCentroids> knn,
                         EmbeddingScoringMethod scoring_method)
        : its_(std::move(its)), context_collector(std::move(context_collector)), knn_(std::move(knn)), scoring_method(scoring_method) {
    for (int i = (its_.size()) - 1; i >= 0; --i) {
        heapify(i);
    }

    last_doc_id_ =
            (its_.empty() || !its_[0]->is_valid()) ? -1 : its_[0]->doc_id();
}

void ANNIterator::advance() {
    if (its_.empty() || !its_[0]->is_valid()) {
        return;
    }

    do {
        its_[0]->advance();
        heapify(0);
        if (!its_[0]->is_valid()) {
            std::swap(its_[0], its_.back());
            its_.pop_back();
            heapify(0);
        }
    } while (!its_.empty() && its_[0]->is_valid() &&
             its_[0]->doc_id() == last_doc_id_);

    if (!its_.empty() && its_[0]->is_valid()) {
        last_doc_id_ = its_[0]->doc_id();
    }
}

bool ANNIterator::is_valid() {
    return !its_.empty() && its_[0]->is_valid();
}

idx_t ANNIterator::doc_id() const {
    return its_[0]->doc_id();
}

std::vector<DocValue> ANNIterator::fields() const {
    std::vector<DocValue> combined_fields;
    idx_t current_doc_id = doc_id();
    for (const auto& it : its_) {
        if (it->is_valid() && it->doc_id() == current_doc_id) {
            auto doc_fields = it->fields();
            combined_fields.insert(
                    combined_fields.end(),
                    doc_fields.begin(),
                    doc_fields.end());
        }
    }

    auto context_fields = this->context_collector.get_context_values(this->doc_id());
    for(const auto &context : context_fields) {
        combined_fields.push_back(context);
    }
    return combined_fields;
}

void ANNIterator::heapify(size_t idx) {
    size_t count_ = its_.size();
    size_t min = idx;
    do {
        idx = min;
        size_t left = idx << 1;
        size_t right = left + 1;
        if (left < count_ && its_[left]->is_valid() &&
            its_[left]->doc_id() < its_[min]->doc_id()) {
            min = left;
        }
        if (right < count_ && its_[right]->is_valid() &&
            its_[right]->doc_id() < its_[min]->doc_id()) {
            min = right;
        }
        if (min != idx) {
            std::swap(its_[idx], its_[min]);
        }
    } while (min != idx);
}

ScoredDocument ANNIterator::score(std::vector<DocValue> fields) const {
    score_t score = lintdb::score_embeddings(this->scoring_method, fields, this->knn_);
    return ScoredDocument(score, 0, fields);
}

AndIterator::AndIterator(std::vector<std::unique_ptr<DocIterator>> iterators,
                         NaryScoringMethod scoring_method)
        : its_(std::move(iterators)), current_doc_id_(0), is_valid_(true), scoring_method(scoring_method) {
    if (its_.empty()) {
        is_valid_ = false;
    } else {
        synchronize(); // Attempt to synchronize on construction
    }
}

void AndIterator::synchronize() {
    if (its_.empty()) {
        is_valid_ = false;
        return;
    }

    bool all_aligned;
    do {
        idx_t max_doc_id = std::numeric_limits<idx_t>::min();
        all_aligned = true;

        // Find the maximum current document ID among all iterators
        for (const auto& it : its_) {
            if (it->is_valid() && it->doc_id() > max_doc_id) {
                max_doc_id = it->doc_id();
            }
        }

        // Attempt to advance any iterators that are behind
        for (auto& it : its_) {
            while (it->is_valid() && it->doc_id() < max_doc_id) {
                it->advance();
                if (it->doc_id() > max_doc_id) {
                    max_doc_id = it->doc_id();
                    all_aligned = false;
                    break;
                }
            }
            // Check if all iterators are still valid
            if (!it->is_valid() || it->doc_id() != max_doc_id) {
                all_aligned = false;
                break;
            }
        }
    } while (!all_aligned &&
             std::all_of(its_.begin(), its_.end(), [](const auto& it) {
                 return it->is_valid();
             }));

    if (all_aligned && !its_.empty()) {
        current_doc_id_ = its_.front()->doc_id();
        is_valid_ = true;
    } else {
        is_valid_ = false;
    }
}

void AndIterator::advance() {
    if (!is_valid_)
        return;

    for (auto& it : its_) {
        if (it->is_valid()) {
            it->advance();
        } else {
            is_valid_ = false;
            return;
        }
    }
    synchronize(); // Realign after advancing
}

bool AndIterator::is_valid() {
    return is_valid_;
}

idx_t AndIterator::doc_id() const {
    return current_doc_id_;
}

std::vector<DocValue> AndIterator::fields() const {
    std::vector<DocValue> results;
    if (is_valid_) {
        for (const auto& it : its_) {
            const std::vector<DocValue>& fields = it->fields();
            results.insert(results.end(), fields.begin(), fields.end());
        }
    }
    return results;
}
ScoredDocument AndIterator::score(std::vector<DocValue> fields) const {
    std::vector<score_t> scores;
    for (const auto& it : its_) {
        auto scored_doc = it->score(fields);
        scores.push_back(scored_doc.score);
    }

    score_t score = lintdb::score(this->scoring_method, scores);
    return ScoredDocument(score, 0, fields);
}

OrIterator::OrIterator(std::vector<std::unique_ptr<DocIterator>> its,
                       NaryScoringMethod scoring_method)
        : its_(std::move(its)), scoring_method(scoring_method) {
    for (int i = (its_.size()) - 1; i >= 0; --i) {
        heapify(i);
    }

    last_doc_id_ =
            (its_.empty() || !its_[0]->is_valid()) ? -1 : its_[0]->doc_id();
}

void OrIterator::advance() {
    if (its_.empty() || !its_[0]->is_valid()) {
        return;
    }

    do {
        its_[0]->advance();
        heapify(0);
        if (!its_[0]->is_valid()) {
            std::swap(its_[0], its_.back());
            its_.pop_back();
            heapify(0);
        }
    } while (!its_.empty() && its_[0]->is_valid() &&
             its_[0]->doc_id() == last_doc_id_);

    if (!its_.empty() && its_[0]->is_valid()) {
        last_doc_id_ = its_[0]->doc_id();
    }
}

bool OrIterator::is_valid() {
    return !its_.empty() && its_[0]->is_valid();
}

idx_t OrIterator::doc_id() const {
    return its_[0]->doc_id();
}

std::vector<DocValue> OrIterator::fields() const {
    std::vector<DocValue> combined_fields;
    idx_t current_doc_id = doc_id();
    for (const auto& it : its_) {
        if (it->is_valid() && it->doc_id() == current_doc_id) {
            auto doc_fields = it->fields();
            combined_fields.insert(
                    combined_fields.end(),
                    doc_fields.begin(),
                    doc_fields.end());
        }
    }
    return combined_fields;
}

void OrIterator::heapify(size_t idx) {
    size_t count_ = its_.size();
    size_t min = idx;
    do {
        idx = min;
        size_t left = idx << 1;
        size_t right = left + 1;
        if (left < count_ && its_[left]->is_valid() &&
            its_[left]->doc_id() < its_[min]->doc_id()) {
            min = left;
        }
        if (right < count_ && its_[right]->is_valid() &&
            its_[right]->doc_id() < its_[min]->doc_id()) {
            min = right;
        }
        if (min != idx) {
            std::swap(its_[idx], its_[min]);
        }
    } while (min != idx);
}

ScoredDocument OrIterator::score(std::vector<DocValue> fields) const {
    std::vector<score_t> scores;
    for (const auto& it : its_) {
        auto scored_doc = it->score(fields);
        scores.push_back(scored_doc.score);
    }

    score_t score = lintdb::score(this->scoring_method, scores);
    return ScoredDocument(score, 0, fields);
}
} // namespace lintdb