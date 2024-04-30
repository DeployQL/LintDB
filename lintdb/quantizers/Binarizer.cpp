#include "lintdb/quantizers/Binarizer.h"
#include <faiss/utils/hamming.h>
#include <glog/logging.h>
#include <json/json.h>
#include <json/reader.h>
#include <json/writer.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include "lintdb/assert.h"
#include "lintdb/util.h"

namespace lintdb {
Binarizer::Binarizer(size_t nbits, size_t dim)
        : Quantizer(), nbits(nbits), dim(dim) {
    LINTDB_THROW_IF_NOT_FMT(
            dim % 8 == 0, "Dimension must be a multiple of 8, got %d", dim);
    LINTDB_THROW_IF_NOT_FMT(
            dim % (nbits * 8) == 0,
            "Dimension must be a multiple of %d, got %d",
            nbits * 8,
            dim);
}

void Binarizer::train(size_t n, const float* x, size_t dim) {
    LOG(INFO) << "Training binarizer with " << n << " vectors of dimension "
              << dim << " and " << nbits << " bits.";
    std::vector<float> avg_residual(dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            avg_residual[j] += *(x + i * dim + j);
        }
    }
    for (size_t j = 0; j < dim; ++j) {
        avg_residual[j] /= n;
    }

    calculate_quantiles(avg_residual);

    float total_avg = 0;
    for (size_t i = 0; i < dim; i++) {
        total_avg += avg_residual[i];
    }
    total_avg /= dim;

    this->avg_residual = total_avg;

    reverse_bitmap = create_reverse_bitmap();
    decompression_lut = create_decompression_lut();
}

void Binarizer::set_weights(
        const std::vector<float>& weights,
        const std::vector<float>& cutoffs,
        const float avg_residual) {
    LINTDB_THROW_IF_NOT(weights.size() == 1 << nbits);

    this->bucket_weights = weights;
    this->bucket_cutoffs = cutoffs;
    this->avg_residual = avg_residual;
    this->reverse_bitmap = create_reverse_bitmap();
    this->decompression_lut = create_decompression_lut();
}

QuantizerType Binarizer::get_type() {
    return QuantizerType::BINARIZER;
}

void Binarizer::save(std::string path) {
    Json::Value root;

    // Fill JSON object with struct data
    root["bucket_cutoffs"] = Json::arrayValue;
    for (const auto& value : bucket_cutoffs) {
        root["bucket_cutoffs"].append(value);
    }

    root["bucket_weights"] = Json::arrayValue;
    for (const auto& value : bucket_weights) {
        root["bucket_weights"].append(value);
    }

    root["avg_residual"] = avg_residual;
    root["nbits"] = static_cast<int>(nbits);
    root["dim"] = static_cast<int>(dim);

    root["reverse_bitmap"] = Json::arrayValue;
    for (const auto& value : reverse_bitmap) {
        root["reverse_bitmap"].append(value);
    }

    root["decompression_lut"] = Json::arrayValue;
    for (const auto& value : decompression_lut) {
        root["decompression_lut"].append(value);
    }

    // Write JSON object to file
    std::ofstream out(path + "/" + QUANTIZER_FILENAME);
    Json::StyledWriter writer;
    if (out.is_open()) {
        out << writer.write(root);
        out.close();
    } else {
        LOG(ERROR) << "Unable to open file for writing: "
                   << path + "/" + QUANTIZER_FILENAME;
    }
}

std::unique_ptr<Binarizer> Binarizer::load(std::string path) {
    // Read JSON file
    std::ifstream file(path + "/" + QUANTIZER_FILENAME);
    if (!file.is_open()) {
        LOG(ERROR) << "Unable to open file for writing: "
                   << path + "/" + QUANTIZER_FILENAME;
        return nullptr;
    }

    Json::Value root;
    file >> root;
    file.close();

    std::vector<float> bucket_cutoffs;
    std::vector<float> bucket_weights;
    float avg_residual;
    size_t nbits;
    size_t dim;
    std::vector<uint8_t> reverse_bitmap;
    std::vector<uint8_t> decompression_lut;

    // Parse JSON data and populate struct fields
    const Json::Value& bucket_cutoffs_value = root["bucket_cutoffs"];
    for (const auto& value : bucket_cutoffs_value) {
        bucket_cutoffs.push_back(value.asFloat());
    }

    const Json::Value& bucket_weights_value = root["bucket_weights"];
    for (const auto& value : bucket_weights_value) {
        bucket_weights.push_back(value.asFloat());
    }

    avg_residual = root["avg_residual"].asFloat();
    nbits = root["nbits"].asUInt();
    dim = root["dim"].asUInt();

    const Json::Value& reverse_bitmap_value = root["reverse_bitmap"];
    for (const auto& value : reverse_bitmap_value) {
        reverse_bitmap.push_back(value.asUInt());
    }

    const Json::Value& decompression_lut_value = root["decompression_lut"];
    for (const auto& value : decompression_lut_value) {
        decompression_lut.push_back(value.asUInt());
    }

    std::unique_ptr<Binarizer> binarizer =
            std::make_unique<Binarizer>(nbits, dim);
    binarizer->bucket_cutoffs = bucket_cutoffs;
    binarizer->bucket_weights = bucket_weights;
    binarizer->avg_residual = avg_residual;
    binarizer->reverse_bitmap = reverse_bitmap;
    binarizer->decompression_lut = decompression_lut;

    return binarizer;
}

void Binarizer::calculate_quantiles(
        const std::vector<float>& heldout_avg_residual) {
    // Calculate average residual and print it
    float sum = 0.0f;
    for (float value : heldout_avg_residual) {
        sum += std::abs(value);
    }
    avg_residual = sum / heldout_avg_residual.size();

    // Calculate quantiles
    int num_options = 1 << nbits;
    std::vector<float> quantiles;
    for (int i = 0; i < num_options; ++i) {
        quantiles.push_back(static_cast<float>(i) / num_options);
    }

    // Calculate bucket cutoffs and weights
    std::vector<float> bucket_cutoffs_quantiles(
            quantiles.begin() + 1, quantiles.end());
    assert(bucket_cutoffs_quantiles.size() == num_options - 1);

    std::vector<float> bucket_weights_quantiles;
    for (float quantile : quantiles) {
        bucket_weights_quantiles.push_back(quantile); // + 0.5f/num_options);
    }

    std::vector<float> sorted_res(heldout_avg_residual);
    std::sort(sorted_res.begin(), sorted_res.end());
    // Quantile function (assuming sorted data)
    auto quantile_func = [&](float quantile) {
        int index = quantile * heldout_avg_residual.size();
        return sorted_res[index];
    };

    std::transform(
            bucket_cutoffs_quantiles.begin(),
            bucket_cutoffs_quantiles.end(),
            std::back_inserter(bucket_cutoffs),
            [&](float quantile) { return quantile_func(quantile); });

    std::transform(
            bucket_weights_quantiles.begin(),
            bucket_weights_quantiles.end(),
            std::back_inserter(bucket_weights),
            [&](float quantile) { return quantile_func(quantile); });
}

// Function to pack bits into big-endian format
std::vector<uint8_t> Binarizer::packbits(
        const std::vector<uint8_t>& binarized) {
    // binarized is sized (dim * nbits)
    std::vector<uint8_t> packed((binarized.size()) / 8, 0);

    for (size_t i = 0; i < binarized.size(); ++i) {
        size_t byteIndex = i / 8;
        size_t bitOffset = i % 8;
        uint8_t bit = binarized[i]; // Get the current bit

        // Determine bit position in big-endian format
        size_t bigEndianBitOffset = 7 - bitOffset;
        // Set the corresponding bit in the packed array
        packed[byteIndex] |= bit << bigEndianBitOffset;
    }

    return packed;
}

// Function to unpack bits from big-endian format
std::vector<uint8_t> Binarizer::unpackbits(
        const std::vector<uint8_t>& packed,
        size_t dim,
        size_t nbits) {
    std::vector<uint8_t> unpacked(dim * nbits, 0);

    for (size_t i = 0; i < dim * nbits; ++i) {
        size_t byteIndex = i / 8;
        size_t bitOffset = i % 8;

        // Determine bit position in big-endian format
        size_t bigEndianBitOffset = 7 - bitOffset;
        uint8_t bit = (packed[byteIndex] >> bigEndianBitOffset) &
                1; // Extract the corresponding bit

        // Set the bit in the unpacked array
        unpacked[i] = bit;
    }

    return unpacked;
}

std::vector<uint8_t> Binarizer::bucketize(const std::vector<float>& residuals) {
    // residuals is a vector of size dim.
    std::vector<uint8_t> binarized(residuals.size() * nbits);

    for (size_t i = 0; i < residuals.size(); ++i) {
        uint8_t bucket = 0;
        bool bucket_found = false;
        for (size_t j = 0; j < bucket_cutoffs.size(); ++j) {
            if (residuals[i] < bucket_cutoffs[j]) {
                bucket = static_cast<uint8_t>(j);
                bucket_found = true;
                break;
            }
        }
        // If the residual is larger than all bucket_cutoffs, assign it to the
        // last bucket
        if (!bucket_found) {
            bucket = static_cast<uint8_t>(bucket_cutoffs.size());
        }

        for (size_t j = 0; j < nbits; ++j) {
            binarized[i * nbits + j] = (bucket >> j) & 1;
        }
    }

    return binarized;
}

std::vector<uint8_t> Binarizer::binarize(const std::vector<float>& residuals) {
    std::vector<uint8_t> binarized = bucketize(residuals);

    return packbits(binarized);
}

std::vector<uint8_t> Binarizer::create_reverse_bitmap() {
    std::vector<uint8_t> reversed_bit_map(256, 0);
    uint8_t mask = (1 << nbits) - 1;

    for (int i = 0; i < 256; ++i) {
        uint8_t z = 0;
        for (int j = 8; j > 0; j -= nbits) {
            uint8_t x = (i >> (j - nbits)) & mask;
            uint8_t y = 0;
            for (int k = nbits - 1; k >= 0; --k) {
                y += ((x >> (nbits - k - 1)) & 1) * pow(2, k);
            }
            z |= y;
            if (j > nbits) {
                z <<= nbits;
            }
        }
        reversed_bit_map[i] = z;
    }

    return reversed_bit_map;
}

std::vector<uint8_t> Binarizer::create_decompression_lut() {
    size_t keys_per_byte = 8 / nbits;
    size_t num_keys = bucket_weights.size();

    std::vector<uint8_t> initial_pool(num_keys, 0);
    std::iota(initial_pool.begin(), initial_pool.end(), 0);
    std::vector<std::vector<uint8_t>> pools = {initial_pool};

    std::vector<uint8_t> lookup_data = product<uint8_t>(pools, keys_per_byte);

    return lookup_data;
}

void Binarizer::sa_encode(size_t n, const float* x, residual_t* codes) {
    for (size_t i = 0; i < n; ++i) {
        // TODO (mbarta): stop making this copy.
        std::vector<float> residuals(x + i * dim, x + (i + 1) * dim);
        std::vector<uint8_t> binarized = binarize(residuals);
        // auto code_size = (nbits + 7) / 8;
        assert(binarized.size() == (dim / 8 * nbits));

        auto code_size = (dim / 8 * nbits);
        for (size_t j = 0; j < code_size; ++j) {
            codes[i * code_size + j] = binarized[j];
        }
    }
}

void Binarizer::sa_decode(size_t n, const residual_t* residuals, float* x) {
    const size_t npacked_vals_per_byte = (8 / nbits);

    const size_t packed_dim = (dim / npacked_vals_per_byte);
    // for each token doc.
    for (size_t i = 0; i < n; ++i) {
        // for each packed residual value
        for (int k = 0; k < packed_dim; ++k) {
            uint8_t packed = residuals[i * packed_dim + k];
            uint8_t reversed_bitmap_val = reverse_bitmap[packed];

            // hydrate each residual value
            for (int l = 0; l < npacked_vals_per_byte; ++l) {
                const int idx = k * npacked_vals_per_byte + l;
                const int bucket_weight_idx = decompression_lut
                        [reversed_bitmap_val * npacked_vals_per_byte + l];

                x[i * dim + idx] = bucket_weights[bucket_weight_idx];
            }
        }
    }
}
} // namespace lintdb