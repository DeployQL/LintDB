#ifndef LINTDB_ENCODERV2_H
#define LINTDB_ENCODERV2_H

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <memory>
#include <string>
#include <vector>
#include "EmbeddingBlock.h"
#include "Passages.h"
#include "SearchOptions.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/quantizers/CoarseQuantizer.h"
#include "lintdb/Encoder.h"

namespace lintdb {
/**
 * DefaultEncoder is a simple encoder that uses L1 quantization to assign codes.
 *
 * That's it.
 */
struct DefaultEncoderV2 : public Encoder {
    size_t dim;                   // number of dimensions per embedding.
    IndexEncoding quantizer_type; // the type of quantizer we encode the
                                  // residuals with.

    std::unique_ptr<CoarseQuantizer> coarse_quantizer;

    DefaultEncoderV2(
            size_t nlist,
            size_t nbits,
            size_t n_iter,
            size_t dim,
            size_t num_subquantizers,
            IndexEncoding type = IndexEncoding::BINARIZER);

    // create a new encoder
    DefaultEncoderV2(
            size_t nbits,
            size_t dim,
            size_t num_subquantizers,
            IndexEncoding type = IndexEncoding::BINARIZER);

    DefaultEncoderV2(size_t dim, std::shared_ptr<Quantizer> quantizer);

    size_t get_dim() const override {
        return dim;
    }

    size_t get_num_centroids() const override {
        return nlist;
    }

    size_t get_nbits() const override {
        return quantizer->get_nbits();
    }

    Quantizer* get_quantizer() const override {
        return quantizer.get();
    }

    std::unique_ptr<EncodedDocument> encode_vectors(
            const EmbeddingPassage& doc) override;

    std::vector<float> decode_vectors(
            gsl::span<const code_t> codes,
            gsl::span<const residual_t> residuals,
            size_t num_tokens) const override;

    void search(
            const float* data,
            const int n,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            const size_t k_top_centroids = 1,
            const float centroid_threshold = 0.45) override;

    void search_quantizer(
            const float* data, // size: (num_query_tok, dim)
            const int num_query_tok,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            const size_t k_top_centroids,
            const float centroid_threshold) override;

    float* get_centroids() const override;

    static std::unique_ptr<Encoder> load(
            std::string path,
            std::shared_ptr<Quantizer> quantizer,
            EncoderConfig& config);
    void train(
            const float* embeddings,
            const size_t n,
            const size_t dim,
            const int n_list = 0,
            const int n_iter = 10) override;

    /**
     * set_centroids overwrites the centroids in the encoder.
     *
     * This is useful if you want to parallelize index writing and merge indices
     * later.
     */
    void set_centroids(float* data, int n, int dim) override;
    void set_weights(
            const std::vector<float>& weights,
            const std::vector<float>& cutoffs,
            const float avg_residual) override;

   private:
    std::shared_ptr<Quantizer> quantizer;
    void save(std::string path) override;
};
} // namespace lintdb

#endif