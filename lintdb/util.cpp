#include "lintdb/util.h"
#include <glog/logging.h>
#include <fstream>
#include <unordered_map>
#include "lintdb/api.h"
#include "lintdb/exception.h"
#include "lintdb/SearchOptions.h"

namespace lintdb {
extern "C" {
// this is to keep the clang syntax checker happy
#ifndef FINTEGER
#define FINTEGER int
#endif

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

float cblas_snrm2(FINTEGER n, const float* x, FINTEGER incx);

int cblas_sscal(FINTEGER n, const float alpha, float* x, FINTEGER incx);
}

void normalize_vector(
        float* doc_residuals,
        const size_t num_doc_tokens,
        const size_t dim) {
    float mod = 0.0;

    int dim2 = dim;

    for (size_t i = 0; i < num_doc_tokens; i++) {
        mod = cblas_snrm2(dim2, doc_residuals + i * dim2, 1);
        if (mod == 1.0) {
            continue;
        }

        int dim2 = dim;
        float mod2 = 1.0 / mod;
        int incx = 1;
        // auto adjusted = std::max(mod, 1e-12f);
        cblas_sscal(dim2, mod2, doc_residuals + i * dim, incx);
    }
}

Json::Value loadJson(const std::string& path) {
    Json::Value root;
    std::ifstream in(path);
    Json::CharReaderBuilder readerBuilder;
    std::string errs;
    if (in.is_open()) {
        if (!Json::parseFromStream(readerBuilder, in, &root, &errs)) {
            LOG(ERROR) << "Failed to parse JSON from file: " << path
                       << ", Error: " << errs;
        }
        in.close();
    } else {
        LOG(ERROR) << "Unable to open file for reading: " << path;
    }

    return root;
}
} // namespace lintdb