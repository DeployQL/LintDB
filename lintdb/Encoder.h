#ifndef LINTDB_ENCODER_H
#define LINTDB_ENCODER_H

#include <vector>
#include "lintdb/RawPassage.h"
#include "lintdb/api.h"
#include "lintdb/invlists/EncodedDocument.h"
#include <string>
#include <memory>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include "lintdb/Binarizer.h"

namespace lintdb {
    static const std::string QUANTIZER_FILENAME = "_quantizer.bin";

    struct EncoderConfig {
        size_t nlist;
        size_t nbits;
        size_t niter;
        size_t dim;
        bool use_compression;
    };

    struct Encoder {
    public:
        Encoder() = default;
        virtual ~Encoder() = default;

        bool is_trained = false;
            /**
         * Encode vectors translates the embeddings given to us in RawPassage to
         * the internal representation that we expect to see in the inverted lists.
         */
        virtual std::unique_ptr<EncodedDocument> encode_vectors(
                const RawPassage& doc) = 0;

        /**
         * Decode vectors translates out of our internal representation.
         *
         * Note: The interface to this has been changing -- it depends on the
         * structures we use to retrieve the data, which is still in flux.
         */
        virtual std::vector<float> decode_vectors(
                const gsl::span<const code_t> codes,
                const gsl::span<const residual_t> residuals,
                const size_t num_tokens,
                const size_t dim) const = 0;

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
            const size_t k_top_centroids=1,
            const float centroid_threshold=0.45
        ) = 0;

        virtual void search_quantizer(
            const float* data,
            const int n,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            const size_t k_top_centroids=1,
            const float centroid_threshold=0.45
        ) = 0;

        /**
         * score_query returns all dot products for each query token against all centroids.
        */
        virtual std::vector<float> score_query(
            const float* data,
            const int n
        ) = 0;

        virtual std::vector<std::pair<float,idx_t>> rank_centroids(
            const float* data,
            const int n,
            const size_t k_top_centroids=1,
            const float centroid_threshold=0.45
        ) = 0;

        virtual void save(std::string path) = 0;
        virtual void train(const float* embeddings, const size_t n, const size_t dim) = 0;

        virtual void set_centroids(float* data, int n, int dim) = 0;
        virtual void set_weights(const std::vector<float>& weights, const std::vector<float>& cutoffs, const float avg_residual) = 0;
    };

    /**
     * DefaultEncoder is a simple encoder that uses L1 quantization to assign codes.
     * 
     * That's it.
    */
    struct DefaultEncoder : public Encoder {
        size_t nlist; // number of centroids to use in L1 quantizing.
        size_t nbits; // number of bits used in binarizing the residuals.
        size_t niter; // number of iterations to use in k-means clustering.
        size_t dim; // number of dimensions per embedding.
        bool use_compression;

        std::unique_ptr<faiss::IndexFlat> quantizer;
        // create a new encoder
        DefaultEncoder(
            size_t nlist, 
            size_t nbits, 
            size_t niter, 
            size_t dim,
            bool use_compression=false);

        std::unique_ptr<EncodedDocument> encode_vectors(
                const RawPassage& doc) override;

        std::vector<float> decode_vectors(
                gsl::span<const code_t> codes,
                gsl::span<const residual_t> residuals,
                size_t num_tokens,
                size_t dim) const override;

        void search(
            const float* data,
            const int n,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            const size_t k_top_centroids=1,
            const float centroid_threshold=0.45
        ) override;

        void search_quantizer(
            const float* data,
            const int n,
            std::vector<idx_t>& coarse_idx,
            std::vector<float>& distances,
            const size_t k_top_centroids=1,
            const float centroid_threshold=0.45
        ) override;

        std::vector<float> score_query(
            const float* data,
            const int n
        ) override;

        std::vector<std::pair<float,idx_t>> rank_centroids(
            const float* data,
            const int n,
            const size_t k_top_centroids,
            const float centroid_threshold
        ) override;
        
        static std::unique_ptr<Encoder> load(std::string path, EncoderConfig& config);
        void train(const float* embeddings, const size_t n, const size_t dim) override;

        /**
         * set_centroids overwrites the centroids in the encoder.
         * 
         * This is useful if you want to parallelize index writing and merge indices later.
        */
        void set_centroids(float* data, int n, int dim) override;
        void set_weights(const std::vector<float>& weights, const std::vector<float>& cutoffs, const float avg_residual) override;

        private:
        std::unique_ptr<Binarizer> binarizer;
        void save(std::string path) override;
    };
}

#endif