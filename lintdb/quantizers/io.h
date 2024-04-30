#ifndef LINTDB_QUANTIZERS_IO_H
#define LINTDB_QUANTIZERS_IO_H

#include <memory>
#include "lintdb/quantizers/Quantizer.h"
#include "lintdb/quantizers/ProductEncoder.h"
#include "lintdb/exception.h"
#include "lintdb/quantizers/Binarizer.h"
#include <faiss/index_io.h>
#include <unordered_map>
#include <string>
#include "lintdb/SearchOptions.h"

namespace lintdb {
    std::unique_ptr<Quantizer> load_quantizer(std::string path, IndexEncoding type, QuantizerConfig& config);
    void save_quantizer(std::string path, Quantizer* quantizer);
    std::unique_ptr<Quantizer> create_quantizer(IndexEncoding type, QuantizerConfig& config);
    
}

#endif