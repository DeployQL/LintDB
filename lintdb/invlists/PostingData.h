#pragma once

#include <string>
#include <vector>

namespace lintdb {
    struct PostingData{
        std::string key;
        std::string value;
    };

    struct BatchPostingData {
        std::vector<PostingData> inverted;
        PostingData forward; /// A single document has one entry in forward index
        std::vector<PostingData> context;
        std::vector<PostingData> inverted_mapping;
    };
}