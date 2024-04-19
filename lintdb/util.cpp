#include "lintdb/util.h"
#include <unordered_map>
#include "lintdb/SearchOptions.h"
#include "lintdb/api.h"
#include "lintdb/exception.h"

namespace lintdb {
extern "C" {
    // this is to keep the clang syntax checker happy
    #ifndef FINTEGER
    #define FINTEGER int
    #endif

    /* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

    int sgemm_(
            const char* transa,
            const char* transb,
            FINTEGER* m,
            FINTEGER* n,
            FINTEGER* k,
            const float* alpha,
            const float* a,
            FINTEGER* lda,
            const float* b,
            FINTEGER* ldb,
            float* beta,
            float* c,
            FINTEGER* ldc);

    float snrm2_(FINTEGER n, const float* x, FINTEGER incx);

    int sscal_(FINTEGER* n, const float* alpha, float* x, FINTEGER* incx);


}

void normalize_vector(
        float* doc_residuals,
        const size_t num_doc_tokens,
        const size_t dim) {
    float mod = 0.0;

    for (size_t i = 0; i < num_doc_tokens; i++) {
        mod = snrm2_(dim, doc_residuals + i * dim, 1);
        if (mod == 1.0) {
            continue;
        }

        int dim2 = dim;
        float mod2 = 1.0 / mod;
        int incx = 1;
        // auto adjusted = std::max(mod, 1e-12f);
        sscal_(&dim2, &mod2, doc_residuals + i * dim, &incx);
    }
}

std::string serialize_encoding(IndexEncoding type) {
    static const std::unordered_map<IndexEncoding, std::string> typeToString{
            {IndexEncoding::NONE, "NONE"},
            {IndexEncoding::BINARIZER, "BINARIZER"},
            {IndexEncoding::PRODUCT_QUANTIZER, "PRODUCT_QUANTIZER"}};

    auto it = typeToString.find(type);
    if (it != typeToString.end()) {
        return it->second;
    } else {
        // Handle error: Unknown enum value
        return "UNKNOWN";
    }
}

IndexEncoding deserialize_encoding(const std::string& str) {
    static const std::unordered_map<std::string, IndexEncoding> stringToType{
            {"NONE", IndexEncoding::NONE},
            {"BINARIZER", IndexEncoding::BINARIZER},
            {"PRODUCT_QUANTIZER", IndexEncoding::PRODUCT_QUANTIZER}};

    auto it = stringToType.find(str);
    if (it != stringToType.end()) {
        return it->second;
    } else {
        throw LintDBException("Unknown string: " + str);
    }
}
} // namespace lintdb