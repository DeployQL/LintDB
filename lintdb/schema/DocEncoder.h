#pragma once

#include "lintdb/schema/DataTypes.h"
#include "lintdb/schema/ProcessedData.h"
#include "lintdb/invlists/keys.h"
#include "lintdb/invlists/PostingData.h"
#include <map>
#include <string>
#include <vector>

namespace lintdb {

    class DocEncoder {
    public:
        static std::vector<PostingData> encode_inverted_data(const ProcessedData& data, size_t code_size);

        static PostingData encode_forward_data(const std::vector<ProcessedData>& data);

        static PostingData encode_context_data(const ProcessedData& data);

        static std::vector<PostingData> encode_inverted_mapping_data(const ProcessedData& data);

        static SupportedTypes decode_supported_types(std::string& data);

        static std::map<uint8_t, SupportedTypes> decode_forward_data(std::string& data);

        static std::vector<idx_t> decode_inverted_mapping_data(std::string& data);
    };

} // lintdb
