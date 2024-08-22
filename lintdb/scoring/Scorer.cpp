#include "Scorer.h"
#include <glog/logging.h>
#include <algorithm>
#include "lintdb/invlists/InvertedList.h"
#include "lintdb/query/decode.h"
#include "lintdb/schema/DocEncoder.h"
#include "ScoredDocument.h"

namespace lintdb {
ColBERTScorer::ColBERTScorer(const lintdb::QueryContext& context) {}
ScoredDocument ColBERTScorer::score(
        QueryContext& context,
        idx_t doc_id,
        std::vector<DocValue>& dvs) const {
    size_t colbert_data_idx = -1;
    for (size_t i = 0; i < dvs.size(); i++) {
        if (dvs[i].type == DataType::COLBERT) {
            colbert_data_idx = i;
            break;
        }
    }

    if (colbert_data_idx == -1) {
        LOG(WARNING) << "colbert context field not found for doc_id: "
                     << doc_id;
        return {0.0, doc_id, dvs};
    }

    uint8_t colbert_field_id =
            context.getFieldMapper()->getFieldID(context.colbert_context);
    size_t dim = context.getFieldMapper()->getFieldDimensions(colbert_field_id);

    SupportedTypes colbert_data = dvs[colbert_data_idx].value;
    ColBERTContextData colbert = std::get<ColBERTContextData>(colbert_data);

    size_t num_tensors = colbert.doc_codes.size();

    std::shared_ptr<Quantizer> quantizer =
            context.getQuantizer(context.colbert_context);

    // decompress residuals.
    Tensor decompressed(num_tensors * dim, 0);
    quantizer->sa_decode(
            num_tensors, colbert.doc_residuals.data(), decompressed.data());

    QueryTensor query =
            context.getOrCreateNearestCentroids(context.colbert_context)
                    ->get_query_tensor();

    auto query_span = gsl::span<float>(query.query);

    DocumentScore score = score_document_by_residuals(
            query_span,
            query.num_query_tokens,
            decompressed.data(),
            num_tensors,
            dim,
            -1,
            true);

    return {score.score, doc_id, dvs};
}

PlaidScorer::PlaidScorer(const QueryContext& context) {
}

ScoredDocument PlaidScorer::score(
        QueryContext& context,
        idx_t doc_id,
        std::vector<DocValue>& fvs) const {

    int colbert_idx = -1;
    for (size_t i = 0; i < fvs.size(); i++) {
        if (fvs[i].type == DataType::COLBERT) {
            colbert_idx = i;
            break;
        }
    }

    if (colbert_idx == -1) {
        LOG(WARNING) << "plaid context field not found for doc_id: " << doc_id;
        return {0.0, doc_id, fvs};
    }

    // rank phase 1: use the codes to score the document using the centroid
    // scores.
    auto nearest_centroids =
            context.getOrCreateNearestCentroids(context.colbert_context);
    auto reordered_distances = nearest_centroids->get_reordered_distances();

    std::shared_ptr<ICoarseQuantizer> coarse_quantizer =
            context.getCoarseQuantizer(context.colbert_context);
    // decode the quantized vector.
    std::shared_ptr<Quantizer> quantizer =
            context.getQuantizer(context.colbert_context);

    // gives us a potentially quantized vector
    SupportedTypes colbert_context = fvs[colbert_idx].value;
    ColBERTContextData codes = std::get<ColBERTContextData>(colbert_context);
    size_t num_tensors = codes.doc_codes.size();

    QueryTensor query =
            context.getOrCreateNearestCentroids(context.colbert_context)
                    ->get_query_tensor();
    float score = colbert_centroid_score(
            codes.doc_codes,
            reordered_distances,
            query.num_query_tokens,
            coarse_quantizer->num_centroids(),
            -1);
    // end rank phase 1.
    return {score, doc_id, fvs};
}
} // namespace lintdb