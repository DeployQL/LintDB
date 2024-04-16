#include "lintdb/quantizers/io.h"

namespace lintdb {
    std::unique_ptr<Quantizer> load_quantizer(std::string path, IndexEncoding type, QuantizerConfig& config) {
        if (FILE *file = fopen((path + "/" + QUANTIZER_FILENAME).c_str(), "r")) {
            fclose(file);
            switch (type)
            {
            case IndexEncoding::NONE:
                return nullptr;
            case IndexEncoding::BINARIZER:
                return Binarizer::load(path);

            case IndexEncoding::PRODUCT_QUANTIZER:
                return ProductEncoder::load(path, config);
            
            default:
                throw LintDBException("Quantizer type not valid.");
            }
            return ProductEncoder::load(path, config);
        } else if (FILE *file = fopen((path + "/" + LEGACY_QUANTIZER_FILENAME).c_str(), "r")) {
            return Binarizer::load(path);
        } else {
            throw LintDBException("Quantizer not found at path: " + path);
        }
    }

    void save_quantizer(std::string path, Quantizer* quantizer) {
        if (quantizer == nullptr) {
            return;
        }

        switch (quantizer->get_type())
        {
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

    std::unique_ptr<Quantizer> create_quantizer(IndexEncoding type, QuantizerConfig& config) {
        switch (type)
        {
        case IndexEncoding::NONE:
            return nullptr;

        case IndexEncoding::BINARIZER:
            return std::make_unique<Binarizer>(config.nbits, config.dim);
        
        case IndexEncoding::PRODUCT_QUANTIZER:
            return std::make_unique<ProductEncoder>(config.dim, config.nbits, config.num_subquantizers);
        
        default:
            throw LintDBException("Quantizer type not valid.");
        }
    }
}