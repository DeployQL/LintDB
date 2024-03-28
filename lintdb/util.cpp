#include "lintdb/util.h"
#include "lintdb/api.h"
#include <cblas.h>

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
}