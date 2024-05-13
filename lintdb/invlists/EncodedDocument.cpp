#include "lintdb/invlists/EncodedDocument.h"
#include <string_view>
#include <memory>
#include <bitsery/bitsery.h>
#include <bitsery/adapter/buffer.h>
#include <bitsery/traits/vector.h>
#include <bitsery/traits/string.h>
#include <bitsery/ext/std_map.h>
#include <glog/logging.h>
#include <map>

namespace bitsery {
    template<typename S>
    void serialize(S& s, std::map<std::string, std::string>& m) {
        s.ext(m, bitsery::ext::StdMap{0xFFF}, [](S& s, std::string& key, std::string& value) {
            s.text1b(key, 0xFF);
            s.text1b(value, 0xFFFF);
        });
    }
}

namespace lintdb {

EncodedDocument::EncodedDocument(
        const std::vector<code_t>
                c, // reflects the centroid id for each token vector.
        const std::vector<residual_t>
                r, // reflects the residual vector for each token vector.
        size_t num_tokens,
        idx_t id,
        const std::map<std::string, std::string>& metadata)
        : codes(c), residuals(r), num_tokens(num_tokens), id(id), metadata(metadata) {}

EncodedDocument::EncodedDocument(
        const code_t* c, // reflects the centroid id for each token vector.
        const size_t codes_size,
        const uint8_t* r, // reflects the residual vector for each token vector.
        const size_t residuals_size,
        size_t num_tokens,
        idx_t id,
        const std::map<std::string, std::string>& metadata)
        : codes(c, c + codes_size),
          residuals(r, r + residuals_size),
          num_tokens(num_tokens),
          id(id),
          metadata(metadata) {}

std::string EncodedDocument::serialize_metadata() const {
    using Buffer = std::vector<uint8_t>;
    using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;

    Buffer buf;
    // TODO(mbarta): DocumentMetadata makes an unnecessary copy here.
    auto written = bitsery::quickSerialization(OutputAdapter{buf}, metadata);
    auto st = std::string(buf.begin(), buf.begin()+written);
    return st;
}

std::unique_ptr<DocumentMetadata> DocumentMetadata::deserialize(std::string& metadata) {
    if (metadata.size() <= 1) {
        return std::make_unique<DocumentMetadata>(std::map<std::string, std::string>());
    }
    using InputAdapter = bitsery::InputBufferAdapter<std::string>;
    std::map<std::string, std::string> md_obj;
    auto state = bitsery::quickDeserialization(InputAdapter{metadata.begin(), metadata.size()}, md_obj);

    assert(state.first == bitsery::ReaderError::NoError && state.second);

    return std::make_unique<DocumentMetadata>(md_obj);
}



} // namespace lintdb