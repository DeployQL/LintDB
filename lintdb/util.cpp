#include "lintdb/util.h"
#include "lintdb/api.h"
#include <cblas.h>
#include "lintdb/SearchOptions.h"
#include <unordered_map>

namespace lintdb {
    void normalize_vector(float* doc_residuals, const size_t num_doc_tokens, const size_t dim) {
    float mod = 0.0;

    for (size_t i = 0; i < num_doc_tokens; i++) {
        mod = cblas_snrm2(dim, doc_residuals + i * dim, 1);
        if (mod == 1.0) {
            continue;
        }
        // auto adjusted = std::max(mod, 1e-12f);
        cblas_sscal(dim, 1.0 / mod, doc_residuals + i * dim, 1);
    }
    }

    std::string serialize_encoding(IndexEncoding type) {
        static const std::unordered_map<IndexEncoding, std::string> typeToString {
            {IndexEncoding::NONE, "NONE"},
            {IndexEncoding::BINARIZER, "BINARIZER"},
            {IndexEncoding::PRODUCT_QUANTIZER, "PRODUCT_QUANTIZER"}
        };

        auto it = typeToString.find(type);
        if (it != typeToString.end()) {
            return it->second;
        } else {
            // Handle error: Unknown enum value
            return "UNKNOWN";
        }
    }

    IndexEncoding deserialize_encoding(const std::string& str) {
        static const std::unordered_map<std::string, IndexEncoding> stringToType {
            {"NONE", IndexEncoding::NONE},
            {"BINARIZER", IndexEncoding::BINARIZER},
            {"PRODUCT_ENCODER", IndexEncoding::PRODUCT_QUANTIZER}
        };

        auto it = stringToType.find(str);
        if (it != stringToType.end()) {
            return it->second;
        } else {
            // Handle error: Unknown string
            return IndexEncoding::NONE; // or any other appropriate error handling
        }
    }
}