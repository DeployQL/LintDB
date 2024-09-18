#include "lintdb/quantizers/CoarseQuantizer.h"
#include <faiss/Clustering.h>
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <glog/logging.h>
#include "lintdb/quantizers/impl/kmeans.h"

namespace lintdb {
CoarseQuantizer::CoarseQuantizer(size_t d) : d(d) {}

CoarseQuantizer::CoarseQuantizer(
        size_t d,
        const std::vector<float>& centroids,
        size_t k)
        : d(d), k(k), centroids(centroids) {
    is_trained_ = true;
}

void CoarseQuantizer::train(
        const size_t n,
        const float* x,
        size_t k,
        size_t num_iter) {
    this->k = k;
    centroids = kmeans(x, n, d, k, Metric::INNER_PRODUCT, num_iter);

    centroids = std::vector<float>(centroids.data(), centroids.data() + k * d);
    is_trained_ = true;
}

void CoarseQuantizer::save(const std::string& path) {
    serialize(path);
}

void CoarseQuantizer::assign(size_t n, const float* x, idx_t* codes) {
    if (!is_trained()) {
        LOG(INFO) << "centroids size: " << centroids.size();
        LOG(INFO) << "k: " << k;
        throw std::runtime_error("Coarse quantizer is not trained.");
    }
    for (size_t i = 0; i < n; ++i) {
        gsl::span<const float> vec(x + i * d, d);
        codes[i] = find_nearest_centroid_index(vec);
    }
}

void CoarseQuantizer::sa_decode(size_t n, const idx_t* codes, float* x) {
    if (!is_trained()) {
        throw std::runtime_error("Coarse quantizer is not trained.");
    }
    for (size_t i = 0; i < n; ++i) {
        const float* centroid = centroids.data() + codes[i] * d;
        std::copy(centroid, centroid + d, x + i * d);
    }
}

void CoarseQuantizer::compute_residual(
        const float* vec,
        float* residual,
        idx_t centroid_id) {
    if (!is_trained()) {
        throw std::runtime_error("Coarse quantizer is not trained.");
    }
    const float* centroid = centroids.data() + centroid_id * d;
    for (size_t i = 0; i < d; ++i) {
        residual[i] = vec[i] - centroid[i];
    }
}

void CoarseQuantizer::compute_residual_n(
        int n,
        const float* vec,
        float* residual,
        idx_t* centroid_ids) {
    if (!is_trained()) {
        throw std::runtime_error("Coarse quantizer is not trained.");
    }
    for (int i = 0; i < n; ++i) {
        const float* centroid = centroids.data() + centroid_ids[i] * d;
        for (size_t j = 0; j < d; ++j) {
            residual[i * d + j] = vec[i * d + j] - centroid[j];
        }
    }
}

void CoarseQuantizer::reconstruct(idx_t centroid_id, float* embedding) {
    if (!is_trained()) {
        throw std::runtime_error("Coarse quantizer is not trained.");
    }
    const float* centroid = centroids.data() + centroid_id * d;
    std::copy(centroid, centroid + d, embedding);
}

void CoarseQuantizer::search(
        size_t num_query_tok,
        const float* data,
        size_t k_top_centroids,
        float* distances,
        idx_t* coarse_idx) {
    if (!is_trained()) {
        throw std::runtime_error("Coarse quantizer is not trained.");
    }
    for (size_t i = 0; i < num_query_tok; ++i) {
        gsl::span<const float> vec(data + i * d, d);
        std::vector<std::pair<float, size_t>> distance_index_pairs(k);
        for (size_t j = 0; j < k; ++j) {
            gsl::span<const float> centroid(centroids.data() + j * d, d);
            float dist = inner_product(
                    vec, centroid); // Change to euclidean_distance if needed
            distance_index_pairs[j] = std::make_pair(dist, j);
        }

        std::partial_sort(
                distance_index_pairs.begin(),
                distance_index_pairs.begin() + k_top_centroids,
                distance_index_pairs.end(),
                [](const std::pair<float, size_t>& a,
                   const std::pair<float, size_t>& b) {
                    return a.first >
                            b.first; // Change to < for Euclidean distance
                });

        for (size_t j = 0; j < k_top_centroids; ++j) {
            distances[i * k_top_centroids + j] = distance_index_pairs[j].first;
            coarse_idx[i * k_top_centroids + j] =
                    distance_index_pairs[j].second;
        }
    }
}

void CoarseQuantizer::reset() {
    this->centroids.clear();
}

void CoarseQuantizer::add(int n, float* data) {
    this->centroids.insert(this->centroids.end(), data, data + n * d);
}

size_t CoarseQuantizer::code_size() {
    return sizeof(residual_t);
}

size_t CoarseQuantizer::num_centroids() {
    if (!is_trained()) {
        throw std::runtime_error(
                "Coarse quantizer is not trained. Unknown number of centroids.");
    }
    return this->k;
}

float* CoarseQuantizer::get_xb() {
    if (!is_trained()) {
        throw std::runtime_error("Coarse quantizer is not trained.");
    }
    return centroids.data();
}

// Serialize to binary file
void CoarseQuantizer::serialize(const std::string& filename) const {
    std::ofstream ofs;
    ofs.open(filename, std::fstream::out | std::fstream::trunc);
    ofs.exceptions(
            std::ofstream::failbit | std::ofstream::badbit |
            std::ofstream::eofbit);
    if (!ofs || !ofs.is_open()) {
        throw std::runtime_error("Unable to open file for writing.");
    }

    // Write dimensionality and number of centroids
    ofs.write(reinterpret_cast<const char*>(&d), sizeof(d));
    ofs.write(reinterpret_cast<const char*>(&k), sizeof(k));
    ofs.write(
            reinterpret_cast<const char*>(&is_trained_), sizeof(is_trained()));

    // Write centroids
    ofs.write(
            reinterpret_cast<const char*>(centroids.data()),
            k * d * sizeof(float));
    ofs.flush();
    ofs.close();
}

// Deserialize from binary file
std::unique_ptr<CoarseQuantizer> CoarseQuantizer::deserialize(
        const std::string& filename,
        const Version& version) {
    if (version.major == 0 && version.minor <= 4 && version.revision <= 0) {
        auto qptr = std::unique_ptr<faiss::Index>(
                faiss::read_index(filename.c_str()));
        auto faiss_quantizer = std::unique_ptr<faiss::IndexFlat>(
                static_cast<faiss::IndexFlat*>(qptr.release()));

        auto coarse_quantizer =
                std::make_unique<CoarseQuantizer>(faiss_quantizer->d);
        coarse_quantizer->k = faiss_quantizer->ntotal;
        coarse_quantizer->is_trained_ = true;
        coarse_quantizer->centroids = std::vector<float>(
                faiss_quantizer->ntotal * faiss_quantizer->d);
        coarse_quantizer->centroids.assign(
                faiss_quantizer->get_xb(),
                faiss_quantizer->get_xb() +
                        faiss_quantizer->ntotal * faiss_quantizer->d);
        return coarse_quantizer;
    }

    // try catch calling ifs
    try {
        std::ifstream ifs(filename, std::fstream::in);
        ifs.exceptions(
                std::ifstream::failbit | std::ifstream::badbit |
                std::ifstream::eofbit);
        if (!ifs.is_open()) {
            throw std::runtime_error("Unable to open file for reading.");
        }

        size_t d, k;
        bool is_trained;

        // Read dimensionality and number of centroids
        ifs.read(reinterpret_cast<char*>(&d), sizeof(d));
        ifs.read(reinterpret_cast<char*>(&k), sizeof(k));
        ifs.read(reinterpret_cast<char*>(&is_trained), sizeof(is_trained_));

        std::unique_ptr<CoarseQuantizer> cq =
                std::make_unique<CoarseQuantizer>(d);
        cq->k = k;
        cq->is_trained_ = is_trained;
        cq->centroids = std::vector<float>(k * d);
        // Read centroids
        ifs.read(
                reinterpret_cast<char*>(cq->centroids.data()),
                k * d * sizeof(float));
        ifs.close();
        return cq;

    } catch (const std::exception& e) {
        LOG(INFO) << e.what();
    }

    return nullptr;
}

uint8_t CoarseQuantizer::find_nearest_centroid_index(
        gsl::span<const float> vec) const {
    float max_product = -std::numeric_limits<float>::max();
    uint8_t best_index = 0;

    for (size_t i = 0; i < k; ++i) {
        gsl::span<const float> centroid(centroids.data() + i * d, d);
        float product = inner_product(vec, centroid);
        if (product > max_product) {
            max_product = product;
            best_index = static_cast<uint8_t>(i);
        }
    }

    return best_index;
}

FaissCoarseQuantizer::FaissCoarseQuantizer(size_t d) : d(d) {
    index = faiss::IndexFlatIP(d);
}
FaissCoarseQuantizer::FaissCoarseQuantizer(
        size_t d,
        const std::vector<float>& centroids,
        size_t k)
        : d(d), k(k) {
    index = faiss::IndexFlatIP(d);
    index.add(k, centroids.data());
    index.is_trained = true;
    is_trained_ = true;
}

void FaissCoarseQuantizer::train(
        const size_t n,
        const float* x,
        size_t k,
        size_t num_iter) {
    faiss::ClusteringParameters cp;
    cp.niter = num_iter;

    faiss::Clustering clus(d, k, cp);
    clus.train(n, x, index);
    is_trained_ = true;
}
void FaissCoarseQuantizer::save(const std::string& path) {
    faiss::write_index(&index, path.c_str());
}
void FaissCoarseQuantizer::assign(size_t n, const float* x, idx_t* codes) {
    return index.assign(n, x, codes);
}
void FaissCoarseQuantizer::sa_decode(size_t n, const idx_t* codes, float* x) {
    const uint8_t* codes_ptr = reinterpret_cast<const uint8_t*>(codes);
    return index.sa_decode(n, codes_ptr, x);
}
void FaissCoarseQuantizer::compute_residual(
        const float* vec,
        float* residual,
        idx_t centroid_id) {
    return index.compute_residual(vec, residual, centroid_id);
}
void FaissCoarseQuantizer::compute_residual_n(
        int n,
        const float* vec,
        float* residual,
        idx_t* centroid_ids) {
    return index.compute_residual_n(n, vec, residual, centroid_ids);
}
void FaissCoarseQuantizer::reconstruct(idx_t centroid_id, float* embedding) {
    return index.reconstruct(centroid_id, embedding);
}
void FaissCoarseQuantizer::search(
        size_t num_query_tok,
        const float* data,
        size_t k_top_centroids,
        float* distances,
        idx_t* coarse_idx) {
    return index.search(
            num_query_tok, data, k_top_centroids, distances, coarse_idx);
}
void FaissCoarseQuantizer::reset() {
    index.reset();
}
void FaissCoarseQuantizer::add(int n, float* data) {
    index.add(n, data);
}
size_t FaissCoarseQuantizer::code_size() {
    return index.code_size;
}
size_t FaissCoarseQuantizer::num_centroids() {
    return index.ntotal;
}
float* FaissCoarseQuantizer::get_xb() {
    return index.get_xb();
}
void FaissCoarseQuantizer::serialize(const std::string& filename) const {
    faiss::write_index(&index, filename.c_str());
}
std::unique_ptr<FaissCoarseQuantizer> FaissCoarseQuantizer::deserialize(
        const std::string& filename,
        const Version& version) {
    faiss::Index* index = faiss::read_index(filename.c_str());

    auto faiss_quantizer = std::make_unique<FaissCoarseQuantizer>(index->d);
    faiss_quantizer->index = *static_cast<faiss::IndexFlatIP*>(index);

    faiss_quantizer->k = faiss_quantizer->index.ntotal;
    faiss_quantizer->d = faiss_quantizer->index.d;
    faiss_quantizer->is_trained_ = faiss_quantizer->index.is_trained;

    return faiss_quantizer;
}

} // namespace lintdb