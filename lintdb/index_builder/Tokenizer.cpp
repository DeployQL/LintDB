#include "lintdb/index_builder/Tokenizer.h"
#include <fstream>
#include <iostream>
#include <tokenizers_cpp.h>


namespace lintdb {
    Tokenizer::Tokenizer(const std::string& path) {
        auto blob = LoadBytesFromFile(path);
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);
    }

    InputIds Tokenizer::encode(const std::string& text) const {
        return tokenizer->Encode(text);
    }

    std::string Tokenizer::decode(const InputIds& ids) const {
        return tokenizer->Decode(ids);
    }

    std::string LoadBytesFromFile(const std::string& path) {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        if (fs.fail()) {
            std::cerr << "Cannot open " << path << std::endl;
            exit(1);
        }
        std::string data;
        fs.seekg(0, std::ios::end);
        size_t size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);
        data.resize(size);
        fs.read(data.data(), size);
        return data;
    }

    Tokenizer::~Tokenizer() = default;
}