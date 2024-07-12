#include "Scorer.h"
#include "lintdb/invlists/InvertedList.h"
#include <glog/logging.h>

namespace lintdb {
    ColBERTScorer::ColBERTScorer(
            const std::shared_ptr<InvertedList> index,
            uint64_t tenant,
            uint8_t field_id,
            size_t code_size):  field_id(field_id), tenant(tenant), code_size(code_size) {
        this->it = index->get_context_iterator(tenant, field_id);
    }

    float ColBERTScorer::score(idx_t doc_id, std::vector<DocValue>& fvs) const {
        return 0.0;
    }
}