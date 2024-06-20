#include "lintdb/util.h"
#include <unordered_map>
#include "lintdb/SearchOptions.h"
#include "lintdb/api.h"
#include "lintdb/exception.h"
#include "lintdb/utils/math.h"

namespace lintdb {

std::string serialize_encoding(IndexEncoding type) {
    static const std::unordered_map<IndexEncoding, std::string> typeToString{
            {IndexEncoding::NONE, "NONE"},
            {IndexEncoding::BINARIZER, "BINARIZER"},
            {IndexEncoding::PRODUCT_QUANTIZER, "PRODUCT_QUANTIZER"},
            {IndexEncoding::XTR, "XTR"},
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
    static const std::unordered_map<std::string, IndexEncoding> stringToType{
            {"NONE", IndexEncoding::NONE},
            {"BINARIZER", IndexEncoding::BINARIZER},
            {"PRODUCT_QUANTIZER", IndexEncoding::PRODUCT_QUANTIZER},
            {"XTR", IndexEncoding::XTR},
    };

    auto it = stringToType.find(str);
    if (it != stringToType.end()) {
        return it->second;
    } else {
        throw LintDBException("Unknown string: " + str);
    }
}
} // namespace lintdb