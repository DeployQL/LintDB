#include "lintdb/quantizers/impl/kmeans.h"
#include <vector>
#include <random>
#include <gsl/span>
#include "lintdb/assert.h"
#include <glog/logging.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

namespace lintdb {
std::vector<float> kmeans(const float* data, size_t n, size_t dim, size_t k, Metric metric, int iterations) {
    LINTDB_THROW_IF_NOT_MSG(n > k, "Number of data points must be greater than the number of clusters.");

    LOG(INFO) << "clustering " << n << " points in " << dim << " dimensions into " << k << " clusters.";

    faiss::IndexFlatIP index(dim);
    faiss::ClusteringParameters cp;
    cp.niter = iterations;
    cp.nredo = 1;
    cp.verbose = true;
    faiss::Clustering clus(dim, k, cp);

    clus.train(n, data, index);

    return std::vector<float>(index.get_xb(), index.get_xb()+k*dim);
}
}