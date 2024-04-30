#pragma once

/**
 * AVX2 isn't quite ready yet. I'm not the biggest expert in intrinsics, and we're looking to adjust from AVX512 instructions to AVX2.
 * For now, we'll only use the generic library until emvb_avx.h is ready.
*/
#ifdef __AVX2__
#include "lintdb/retriever/emvb_avx.h"

namespace lintdb {
auto filter_query_scores = filter_query_scores_avx;
auto popcount = popcount_avx;
auto compute_ip_with_centroids = compute_ip_with_centroids_avx;
auto compute_score_by_column_reduction = compute_score_by_column_reduction_avx;
auto filter_centroids_in_scoring = filter_centroids_in_scoring_avx;
}

#else
#include "lintdb/retriever/emvb_generic.h"
namespace lintdb {
auto filter_query_scores = filter_query_scores_generic;
auto popcount = popcount_generic;
auto compute_ip_with_centroids = compute_ip_with_centroids_generic;
auto compute_score_by_column_reduction = compute_score_by_column_reduction_generic;
auto filter_centroids_in_scoring = filter_centroids_in_scoring_generic;
}

#endif

