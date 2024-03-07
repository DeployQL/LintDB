#include "lintdb/Encoder.h"

namespace lintdb {
    DefaultEncoder::DefaultEncoder(std::string path) {
        if (FILE *file = fopen((path + "/" + QUANTIZER_FILENAME).c_str(), "r")) {
            fclose(file);
            quantizer = faiss::read_index((path + "/" + QUANTIZER_FILENAME).c_str());
        } else {
            throw LintDBException("Index not found at path: " + path);
        }

        if (FILE *file = fopen((path + "/" + BINARIZER_FILENAME).c_str(), "r")) {
            fclose(file);
            binarizer = faiss::read_index((path + "/" + BINARIZER_FILENAME).c_str());
        } else {
            throw LintDBException("Index not found at path: " + path);
        }
    }

    DefaultEncoder::DefaultEncoder(size_t nlist, size_t nbits, size_t niter, size_t dim)
        : Encoder(), nlist(nlist), nbits(nbits), niter(niter), dim(dim) {
    }

    std::unique_ptr<EncodedDocument> DefaultEncoder::encode_vectors(
            const RawPassage& doc) const {
        return std::make_unique<EncodedDocument>(doc, nlist, nbits, niter, dim);
    }

    std::vector<float> DefaultEncoder::decode_vectors(
            gsl::span<const code_t> codes,
            gsl::span<const residual_t> residuals,
            size_t num_tokens,
            size_t dim) const {
        std::vector<float> decoded;
        decoded.reserve(num_tokens * dim);
        for (size_t i = 0; i < num_tokens; i++) {
            for (size_t j = 0; j < dim; j++) {
                decoded.push_back(residuals[i * dim + j]);
            }
        }
        return decoded;
    }

    void DefaultEncoder::save() {
    }
}