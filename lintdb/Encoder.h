#ifndef LINTDB_ENCODER_H
#define LINTDB_ENCODER_H

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <memory>
#include <string>
#include <vector>
#include "lintdb/Passages.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/api.h"
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/quantizers/Quantizer.h"

namespace lintdb {
static const std::string ENCODER_FILENAME = "_quantizer.bin";

struct EncoderConfig {
    size_t nlist;
    size_t nbits;
    size_t niter;
    size_t dim;
    size_t num_subquantizers;
    IndexEncoding type;
};

// TODO(mbarta): stop using Encoder as an abstract class. It's not necessary.
struct Encoder {
   public:
    Encoder() = default;
    virtual ~Encoder() = default;

    bool is_trained = false;
    size_t dim;
    size_t nlist = 0;
    size_t niter = 0;

    virtual size_t get_dim() const = 0;
    virtual size_t get_num_centroids() const = 0;
    virtual size_t get_nbits() const = 0;
    /**
     * Encode vectors translates the embeddings given to us in EmbeddingPassage to
     * the internal representation that we expect to see in the inverted lists.
     */
    virtual std::unique_ptr<EncodedDocument> encode_vectors(
            const EmbeddingPassage& doc) = 0;

    /**
     * Decode vectors translates out of our internal representation.
     *
     * Note: The interface to this has been changing -- it depends on the
     * structures we use to retrieve the data, which is still in flux.
     */
    virtual std::vector<float> decode_vectors(
            const gsl::span<const code_t> codes,
            const gsl::span<const residual_t> residuals,
            const size_t num_tokens) const = 0;

    /**
     * Given a query, search for the nearest centroids.
     *
     * This has a dual purpose in the index.
     * First, it's used to get the top centroids to search.
     * Second, we use the centroid scores to calculate the first stage of
     * plaid scoring.
     *
     * Because of this dual purpose, the return format is a dense matrix.
     * The caller is responsible for converting it for its purpose.
     */
    virtual void search(
            const float* data,
            const int n,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            const size_t k_top_centroids = 1,
            const float centroid_threshold = 0.45) = 0;
    virtual void search_quantizer(
            const float* data, // size: (num_query_tok, dim)
            const int num_query_tok,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            const size_t k_top_centroids,
            const float centroid_threshold) = 0;

    virtual float* get_centroids() const = 0;
    virtual Quantizer* get_quantizer() const = 0;

    virtual void save(std::string path) = 0;
    virtual void train(
            const float* embeddings,
            const size_t n,
            const size_t dim,
            const int n_list,
            const int n_iter) = 0;

    virtual void set_centroids(float* data, int n, int dim) = 0;
    virtual void set_weights(
            const std::vector<float>& weights,
            const std::vector<float>& cutoffs,
            const float avg_residual) = 0;
};

/**
 * DefaultEncoder is a simple encoder that uses L1 quantization to assign codes.
 *
 * That's it.
 */
struct DefaultEncoder : public Encoder {
    size_t dim;   // number of dimensions per embedding.
    IndexEncoding quantizer_type; // the type of quantizer we encode the
                                  // residuals with.

    std::unique_ptr<faiss::IndexFlat> coarse_quantizer;

    [[deprecated]]
    DefaultEncoder(
            size_t nlist,
            size_t nbits,
            size_t n_iter,
            size_t dim,
            size_t num_subquantizers,
            IndexEncoding type = IndexEncoding::BINARIZER);

    // create a new encoder
    DefaultEncoder(
            size_t nbits,
            size_t dim,
            size_t num_subquantizers,
            IndexEncoding type = IndexEncoding::BINARIZER);

    DefaultEncoder(
            size_t dim,
            std::shared_ptr<Quantizer> quantizer
            );

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
    void train(const float* embeddings, const size_t n, const size_t dim, const int n_list=0, const int n_iter=10)
            override;

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