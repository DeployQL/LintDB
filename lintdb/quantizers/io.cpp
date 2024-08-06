#include "lintdb/quantizers/io.h"
#include <faiss/index_io.h>

namespace lintdb {
std::unique_ptr<Quantizer> load_quantizer(
        std::string path,
        QuantizerType type,
        QuantizerConfig& config) {
    if (type == QuantizerType::NONE) {
        // the file won't exist, so we check NONE first.
        return std::make_unique<IdentityQuantizer>(config.dim);
    }

    if (FILE* file = fopen((path).c_str(), "r")) {
        fclose(file);
        switch (type) {
            case QuantizerType::NONE:
                return std::make_unique<IdentityQuantizer>(config.dim);
            case QuantizerType::BINARIZER:
                return Binarizer::load(path);

            case QuantizerType::PRODUCT_ENCODER:
                return ProductEncoder::load(path, config);

            default:
                throw LintDBException("Quantizer type not valid.");
        }
        return ProductEncoder::load(path, config);
    } else {
        throw LintDBException("Quantizer not found at path: " + path);
    }
}

void save_quantizer(std::string path, Quantizer* quantizer) {
    if (quantizer == nullptr) {
        return;
    }

    switch (quantizer->get_type()) {
        case QuantizerType::NONE:
            break;
        case QuantizerType::BINARIZER:
            quantizer->save(path);
            break;

        case QuantizerType::PRODUCT_ENCODER:
            quantizer->save(path);
            break;

        default:
            throw LintDBException("Quantizer type not valid.");
    }
}

std::unique_ptr<Quantizer> create_quantizer(
        QuantizerType type,
        QuantizerConfig& config) {
    switch (type) {
        case QuantizerType::NONE:
            return std::make_unique<IdentityQuantizer>(config.dim);
            ;

        case QuantizerType::BINARIZER:
            return std::make_unique<Binarizer>(config.nbits, config.dim);

        case QuantizerType::PRODUCT_ENCODER:
            return std::make_unique<ProductEncoder>(
                    config.dim, config.nbits, config.num_subquantizers);

        default:
            throw LintDBException("Quantizer type not valid.");
    }
}
} // namespace lintdb