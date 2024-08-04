#ifndef LINTDB_QUANTIZERS_IO_H
#define LINTDB_QUANTIZERS_IO_H

#include <memory>
#include <string>
#include <unordered_map>
#include "lintdb/exception.h"
#include "lintdb/quantizers/Binarizer.h"
#include "lintdb/quantizers/IdentityQuantizer.h"
#include "lintdb/quantizers/ProductEncoder.h"
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/SearchOptions.h"

namespace lintdb {
std::unique_ptr<Quantizer> load_quantizer(
        std::string path,
        QuantizerType type,
        QuantizerConfig& config);

void save_quantizer(std::string path, Quantizer* quantizer);

std::unique_ptr<Quantizer> create_quantizer(
        QuantizerType type,
        QuantizerConfig& config);
} // namespace lintdb

#endif