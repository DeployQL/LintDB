// automatically generated by the FlatBuffers compiler, do not modify

#ifndef FLATBUFFERS_GENERATED_FORWARDINDEX_LINTDB_H_
#define FLATBUFFERS_GENERATED_FORWARDINDEX_LINTDB_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(
        FLATBUFFERS_VERSION_MAJOR == 23 && FLATBUFFERS_VERSION_MINOR == 5 &&
                FLATBUFFERS_VERSION_REVISION == 26,
        "Non-compatible flatbuffers version included");

namespace lintdb {

struct ForwardIndexDocument;
struct ForwardIndexDocumentBuilder;

struct ForwardIndexDocument FLATBUFFERS_FINAL_CLASS
        : private ::flatbuffers::Table {
    typedef ForwardIndexDocumentBuilder Builder;
    enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
        VT_NUM_TOKENS = 4,
        VT_RESIDUALS = 6
    };
    uint32_t num_tokens() const {
        return GetField<uint32_t>(VT_NUM_TOKENS, 0);
    }
    const ::flatbuffers::Vector<uint8_t>* residuals() const {
        return GetPointer<const ::flatbuffers::Vector<uint8_t>*>(VT_RESIDUALS);
    }
    bool Verify(::flatbuffers::Verifier& verifier) const {
        return VerifyTableStart(verifier) &&
                VerifyField<uint32_t>(verifier, VT_NUM_TOKENS, 4) &&
                VerifyOffset(verifier, VT_RESIDUALS) &&
                verifier.VerifyVector(residuals()) && verifier.EndTable();
    }
};

struct ForwardIndexDocumentBuilder {
    typedef ForwardIndexDocument Table;
    ::flatbuffers::FlatBufferBuilder& fbb_;
    ::flatbuffers::uoffset_t start_;
    void add_num_tokens(uint32_t num_tokens) {
        fbb_.AddElement<uint32_t>(
                ForwardIndexDocument::VT_NUM_TOKENS, num_tokens, 0);
    }
    void add_residuals(
            ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> residuals) {
        fbb_.AddOffset(ForwardIndexDocument::VT_RESIDUALS, residuals);
    }
    explicit ForwardIndexDocumentBuilder(::flatbuffers::FlatBufferBuilder& _fbb)
            : fbb_(_fbb) {
        start_ = fbb_.StartTable();
    }
    ::flatbuffers::Offset<ForwardIndexDocument> Finish() {
        const auto end = fbb_.EndTable(start_);
        auto o = ::flatbuffers::Offset<ForwardIndexDocument>(end);
        return o;
    }
};

inline ::flatbuffers::Offset<ForwardIndexDocument> CreateForwardIndexDocument(
        ::flatbuffers::FlatBufferBuilder& _fbb,
        uint32_t num_tokens = 0,
        ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> residuals = 0) {
    ForwardIndexDocumentBuilder builder_(_fbb);
    builder_.add_residuals(residuals);
    builder_.add_num_tokens(num_tokens);
    return builder_.Finish();
}

inline ::flatbuffers::Offset<ForwardIndexDocument>
CreateForwardIndexDocumentDirect(
        ::flatbuffers::FlatBufferBuilder& _fbb,
        uint32_t num_tokens = 0,
        const std::vector<uint8_t>* residuals = nullptr) {
    auto residuals__ = residuals ? _fbb.CreateVector<uint8_t>(*residuals) : 0;
    return lintdb::CreateForwardIndexDocument(_fbb, num_tokens, residuals__);
}

inline const lintdb::ForwardIndexDocument* GetForwardIndexDocument(
        const void* buf) {
    return ::flatbuffers::GetRoot<lintdb::ForwardIndexDocument>(buf);
}

inline const lintdb::ForwardIndexDocument* GetSizePrefixedForwardIndexDocument(
        const void* buf) {
    return ::flatbuffers::GetSizePrefixedRoot<lintdb::ForwardIndexDocument>(
            buf);
}

inline bool VerifyForwardIndexDocumentBuffer(
        ::flatbuffers::Verifier& verifier) {
    return verifier.VerifyBuffer<lintdb::ForwardIndexDocument>(nullptr);
}

inline bool VerifySizePrefixedForwardIndexDocumentBuffer(
        ::flatbuffers::Verifier& verifier) {
    return verifier.VerifySizePrefixedBuffer<lintdb::ForwardIndexDocument>(
            nullptr);
}

inline void FinishForwardIndexDocumentBuffer(
        ::flatbuffers::FlatBufferBuilder& fbb,
        ::flatbuffers::Offset<lintdb::ForwardIndexDocument> root) {
    fbb.Finish(root);
}

inline void FinishSizePrefixedForwardIndexDocumentBuffer(
        ::flatbuffers::FlatBufferBuilder& fbb,
        ::flatbuffers::Offset<lintdb::ForwardIndexDocument> root) {
    fbb.FinishSizePrefixed(root);
}

} // namespace lintdb

#endif // FLATBUFFERS_GENERATED_FORWARDINDEX_LINTDB_H_
