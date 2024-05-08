#pragma once

#include <vector>
#include <string>
#include <memory>

typedef std::vector<int32_t> InputIds;
typedef std::vector<int32_t> AttentionMask;

namespace tokenizers {
    class Tokenizer;
}

namespace lintdb {
    /**
     * Tokenizer takes a string and returns a list of token ids.
     * 
     * This equates to running a huggingface tokenizer.
    */
    struct Tokenizer {
        Tokenizer(const std::string& path);
        InputIds encode(const std::string& text) const;
        std::string decode(const InputIds& ids) const;

        ~Tokenizer();

        private:
            std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    };

    std::string LoadBytesFromFile(const std::string& path);
}