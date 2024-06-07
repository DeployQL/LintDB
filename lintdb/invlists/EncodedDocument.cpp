#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/api.h"
#include <string_view>
#include <memory>
#include <bitsery/bitsery.h>
#include <bitsery/adapter/buffer.h>
#include <bitsery/traits/vector.h>
#include <bitsery/traits/string.h>
#include <bitsery/ext/std_map.h>
#include <glog/logging.h>
#include <map>
#include <gsl/span>
#include "lintdb/assert.h"

namespace bitsery {
    template<typename S>
    void serialize(S& s, std::map<std::string, std::string>& m) {
        s.ext(m, bitsery::ext::StdMap{0xFFF}, [](S& s, std::string& key, std::string& value) {
            s.text1b(key, 0xFF);
            s.text1b(value, 0xFFFF);
        });
    }

    template<typename S>
    void serialize(S& s, std::vector<residual_t>& v) {
        s.container1b(v, 0xFF);

    }
}

namespace lintdb {

EncodedDocument::EncodedDocument(
        const std::vector<code_t> c,
        const std::vector<residual_t> r,
        size_t num_tokens,
        idx_t id,
        size_t cs,
        const std::map<std::string, std::string>& metadata)
        : codes(c), residuals(r), num_tokens(num_tokens), id(id), metadata(metadata), code_size(cs) {}

EncodedDocument::EncodedDocument(
        const code_t* c,
        const size_t codes_size,
        const uint8_t* r,
        const size_t residuals_size,
        size_t num_tokens,
        idx_t id,
        size_t cs,
        const std::map<std::string, std::string>& metadata)
        : codes(c, c + codes_size),
          residuals(r, r + residuals_size),
          num_tokens(num_tokens),
          id(id),
          metadata(metadata),
          code_size(cs) {}

std::string EncodedDocument::serialize_metadata() const {
    using Buffer = std::vector<uint8_t>;
    using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;

    Buffer buf;
    // TODO(mbarta): DocumentMetadata makes an unnecessary copy here.
    auto written = bitsery::quickSerialization(OutputAdapter{buf}, metadata);
    auto st = std::string(buf.begin(), buf.begin()+written);
    return st;
}

std::vector<InvertedData> EncodedDocument::serialize_inverted_data() const {
    using Buffer = std::vector<residual_t>;
    using OutputAdapter = bitsery::OutputBufferAdapter<Buffer>;

    std::vector<InvertedData> results;

    assert(residuals.size() % code_size == 0);

    for (idx_t i=0; i < codes.size(); i++) {
        // for each code, we want to store the residuals associated with the right
        // tokens.
        auto residuals_start = residuals.begin() + i * code_size;
        std::vector<residual_t> view(residuals_start, residuals_start+code_size);

        InvertedData data;
        data.key = codes[i];
//        Buffer buf;
//        auto written = bitsery::quickSerialization(OutputAdapter{buf}, view);
//        data.value = std::string(buf.begin(), buf.begin()+written);
//        LOG(INFO) << "code size: " << code_size << " residuals size: " << view.size();
        data.value = std::string(residuals_start, residuals_start+code_size);
        data.token_id = i;

        results.push_back(data);
    }


    return results;

}

PartialDocumentCodes PartialDocumentCodes::deserialize(idx_t id, std::string& data) {
    LOG(INFO) << "deserializing partial doc codes. size: " << data.size();

    LINTDB_THROW_IF_NOT(!data.empty());

//    using InputAdapter = bitsery::InputBufferAdapter<std::string>;
//    std::vector<residual_t> residuals;
//    auto state = bitsery::quickDeserialization(InputAdapter{data.begin(), data.size()}, residuals);
//    assert(state.first == bitsery::ReaderError::NoError && state.second);

    std::vector<residual_t> residuals(data.begin(), data.end());

    return PartialDocumentCodes(id, residuals);

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