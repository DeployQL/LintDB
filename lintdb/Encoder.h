#ifndef LINTDB_ENCODER_H
#define LINTDB_ENCODER_H

#include <vector>
#include "lintdb/RawPassage.h"
#include "lintdb/api.h"
#include "lintdb/invlists/EncodedDocument.h"
#include <string>

namespace lintdb {
    static const std::string QUANTIZER_FILENAME = "_quantizer.bin";
    static const std::string BINARIZER_FILENAME = "_binarizer.bin";

    struct Encoder {
    public:
        Encoder() = default;
        virtual ~Encoder() = default;
            /**
         * Encode vectors translates the embeddings given to us in RawPassage to
         * the internal representation that we expect to see in the inverted lists.
         */
        virtual std::unique_ptr<EncodedDocument> encode_vectors(
                const RawPassage& doc) const = 0;

        /**
         * Decode vectors translates out of our internal representation.
         *
         * Note: The interface to this has been changing -- it depends on the
         * structures we use to retrieve the data, which is still in flux.
         */
        virtual std::vector<float> decode_vectors(
                gsl::span<const code_t> codes,
                gsl::span<const residual_t> residuals,
                size_t num_tokens,
                size_t dim) const = 0;

        virtual void save() = 0;
        virtual void train(float* embeddings, size_t n, size_t dim) = 0;
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

        // load an encoder from a database path
        DefaultEncoder(std::string path);

        // create a new encoder
        DefaultEncoder(size_t nlist, size_t nbits, size_t niter, size_t dim);

        std::unique_ptr<EncodedDocument> encode_vectors(
                const RawPassage& doc) const override;

        std::vector<float> decode_vectors(
                gsl::span<const code_t> codes,
                gsl::span<const residual_t> residuals,
                size_t num_tokens,
                size_t dim) const override;

        void save() override;
        void train(float* embeddings, size_t n, size_t dim) override;
    }
}

#endif