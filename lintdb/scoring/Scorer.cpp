#include "Scorer.h"
#include "lintdb/invlists/InvertedList.h"
#include <glog/logging.h>
#include "lintdb/schema/DocEncoder.h"
#include "lintdb/query/decode.h"

namespace lintdb {
    ColBERTScorer::ColBERTScorer(const lintdb::QueryContext &context) {

    }
    ScoredDocument ColBERTScorer::score(QueryContext &context, idx_t doc_id, std::vector<DocValue>& dvs) const {
        size_t colbert_data_idx = -1;
        for (size_t i = 0; i < dvs.size(); i++) {
            if (dvs[i].type == DataType::COLBERT) {
                colbert_data_idx = i;
                break;
            }
        }

        if(colbert_data_idx == -1) {
            LOG(WARNING) << "colbert context field not found for doc_id: " << doc_id;
            return {0.0, doc_id, dvs};
        }

        uint8_t colbert_field_id = context.getFieldMapper()->getFieldID(context.colbert_context);
        size_t dim = context.getFieldMapper()->getFieldDimensions(colbert_field_id);

        SupportedTypes colbert_data = dvs[colbert_data_idx].value;
        ColBERTContextData colbert = std::get<ColBERTContextData>(colbert_data);

        size_t num_tensors = colbert.doc_codes.size();

        std::shared_ptr<Quantizer> quantizer = context.getQuantizer(context.colbert_context);

        // decompress residuals.
        Tensor decompressed(num_tensors * dim, 0);
        quantizer->sa_decode(num_tensors, colbert.doc_residuals.data(), decompressed.data());

        QueryTensor query = context.getOrCreateNearestCentroids(context.colbert_context)->get_query_tensor();

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


    PlaidScorer::PlaidScorer(const QueryContext &context) {
        uint8_t colbert_field_id = context.getFieldMapper()->getFieldID(context.colbert_context);
        colbert_it = context.getIndex()->get_context_iterator(context.getTenant(), colbert_field_id);
    }

    ScoredDocument PlaidScorer::score(QueryContext &context, idx_t doc_id, std::vector<DocValue> &fvs) const {
        colbert_it->advance(doc_id);
        if (!colbert_it->is_valid() || colbert_it->get_key().doc_id() != doc_id) {
            LOG(WARNING) << "plaid context field not found for doc_id: " << doc_id;
            return {0.0, doc_id, fvs};
        }

        // rank phase 1: use the codes to score the document using the centroid scores.
        auto nearest_centroids = context.getOrCreateNearestCentroids(context.colbert_context);
        auto reordered_distances = nearest_centroids->get_reordered_distances();

        std::shared_ptr<ICoarseQuantizer> coarse_quantizer = context.getCoarseQuantizer(context.colbert_context);
        // decode the quantized vector.
        std::shared_ptr<Quantizer> quantizer = context.getQuantizer(context.colbert_context);

        auto context_str = colbert_it->get_value();

        // gives us a potentially quantized vector
        SupportedTypes colbert_context = DocEncoder::decode_supported_types(context_str);
        ColBERTContextData codes = std::get<ColBERTContextData>(colbert_context);
        size_t num_tensors = codes.doc_codes.size();

        // create DocValues for the context info.
        uint8_t colbert_field_id = context.getFieldMapper()->getFieldID(context.colbert_context);
        fvs.emplace_back(colbert_context, colbert_field_id, DataType::COLBERT);

        QueryTensor query = context.getOrCreateNearestCentroids(context.colbert_context)->get_query_tensor();
        float score = colbert_centroid_score(
                codes.doc_codes,
                reordered_distances,
                query.num_query_tokens,
                coarse_quantizer->num_centroids(),
                -1);
        // end rank phase 1.
        return {score, doc_id, fvs};
    }
}