#include "scoring_methods.h"

namespace lintdb {
score_t score_one(const std::vector<DocValue>& values) {
    return 1.0;
}

score_t plaid_similarity(const std::vector<DocValue>& values, std::shared_ptr<KnnNearestCentroids> knn) {
    int colbert_idx = -1;
    for (size_t i = 0; i < values.size(); i++) {
        if (values[i].type == DataType::COLBERT) {
            colbert_idx = i;
            break;
        }
    }

    if (colbert_idx == -1) {
        LOG(WARNING) << "plaid context field not found for doc_id";
        return 0.0;
    }

    // rank phase 1: use the codes to score the document using the centroid
    // scores.
//    auto reordered_distances = knn->get_reordered_distances();

    // gives us a potentially quantized vector
    SupportedTypes colbert_context = values[colbert_idx].value;
    ColBERTContextData codes = std::get<ColBERTContextData>(colbert_context);
    size_t num_tensors = codes.doc_codes.size();

    QueryTensor query = knn->get_query_tensor();
    float score = colbert_centroid_score(
            codes.doc_codes,
            knn->get_reordered_distances(),
            query.num_query_tokens,
            knn->get_num_centroids(),
            -1);
    return score;
}

UnaryScoringMethodFunction unary_scoring_methods[] = {
        score_one,
};

score_t score(const UnaryScoringMethod method, const std::vector<DocValue>& values) {
    int scoring_type = static_cast<int>(method);
    return unary_scoring_methods[scoring_type](values);
}

EmbeddingScoringMethodFunction embedding_scoring_methods[] = {
        plaid_similarity,
};


score_t score_embeddings(const EmbeddingScoringMethod method, const std::vector<DocValue>& values, std::shared_ptr<KnnNearestCentroids> knn) {
    int scoring_type = static_cast<int>(method);
    return embedding_scoring_methods[scoring_type](values, knn);
}

score_t sum(const std::vector<score_t>& values) {
    score_t sum = 0;
    for (const score_t value : values) {
        sum += value;
    }
    return sum;
}

score_t reduce(const std::vector<score_t>& values) {
    score_t product = 1;
    for (const score_t value : values) {
        product *= value;
    }
    return product;
}

score_t max(const std::vector<score_t>& values) {
    score_t max = values[0];
    for (const score_t value : values) {
        if (value > max) {
            max = value;
        }
    }
    return max;
}

NaryScoringMethodFunction nary_scoring_methods[] = {
        sum,
        reduce,
        max,
};

score_t score(const NaryScoringMethod method, const std::vector<score_t>& values) {
    int scoring_type = static_cast<int>(method);
    return nary_scoring_methods[scoring_type](values);
}

}