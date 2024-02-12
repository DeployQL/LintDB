// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_SCHEMA_LINTDB_H_
#define FLATBUFFERS_GENERATED_SCHEMA_LINTDB_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 23 &&
              FLATBUFFERS_VERSION_MINOR == 5 &&
              FLATBUFFERS_VERSION_REVISION == 26,
             "Non-compatible flatbuffers version included");

namespace lintdb {

struct Code;
struct CodeBuilder;

struct InvertedDocument;
struct InvertedDocumentBuilder;

struct Code FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef CodeBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_CODES = 4
  };
  const ::flatbuffers::Vector<uint8_t> *codes() const {
    return GetPointer<const ::flatbuffers::Vector<uint8_t> *>(VT_CODES);
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_CODES) &&
           verifier.VerifyVector(codes()) &&
           verifier.EndTable();
  }
};

struct CodeBuilder {
  typedef Code Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_codes(::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> codes) {
    fbb_.AddOffset(Code::VT_CODES, codes);
  }
  explicit CodeBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<Code> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Code>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<Code> CreateCode(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> codes = 0) {
  CodeBuilder builder_(_fbb);
  builder_.add_codes(codes);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<Code> CreateCodeDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint8_t> *codes = nullptr) {
  auto codes__ = codes ? _fbb.CreateVector<uint8_t>(*codes) : 0;
  return lintdb::CreateCode(
      _fbb,
      codes__);
}

struct InvertedDocument FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef InvertedDocumentBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DOC_ID = 4,
    VT_NUM_TOKENS = 6,
    VT_CODES = 8
  };
  const ::flatbuffers::String *doc_id() const {
    return GetPointer<const ::flatbuffers::String *>(VT_DOC_ID);
  }
  uint32_t num_tokens() const {
    return GetField<uint32_t>(VT_NUM_TOKENS, 0);
  }
  const ::flatbuffers::Vector<uint8_t> *codes() const {
    return GetPointer<const ::flatbuffers::Vector<uint8_t> *>(VT_CODES);
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_DOC_ID) &&
           verifier.VerifyString(doc_id()) &&
           VerifyField<uint32_t>(verifier, VT_NUM_TOKENS, 4) &&
           VerifyOffset(verifier, VT_CODES) &&
           verifier.VerifyVector(codes()) &&
           verifier.EndTable();
  }
};

struct InvertedDocumentBuilder {
  typedef InvertedDocument Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_doc_id(::flatbuffers::Offset<::flatbuffers::String> doc_id) {
    fbb_.AddOffset(InvertedDocument::VT_DOC_ID, doc_id);
  }
  void add_num_tokens(uint32_t num_tokens) {
    fbb_.AddElement<uint32_t>(InvertedDocument::VT_NUM_TOKENS, num_tokens, 0);
  }
  void add_codes(::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> codes) {
    fbb_.AddOffset(InvertedDocument::VT_CODES, codes);
  }
  explicit InvertedDocumentBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<InvertedDocument> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<InvertedDocument>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<InvertedDocument> CreateInvertedDocument(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    ::flatbuffers::Offset<::flatbuffers::String> doc_id = 0,
    uint32_t num_tokens = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>> codes = 0) {
  InvertedDocumentBuilder builder_(_fbb);
  builder_.add_codes(codes);
  builder_.add_num_tokens(num_tokens);
  builder_.add_doc_id(doc_id);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<InvertedDocument> CreateInvertedDocumentDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    const char *doc_id = nullptr,
    uint32_t num_tokens = 0,
    const std::vector<uint8_t> *codes = nullptr) {
  auto doc_id__ = doc_id ? _fbb.CreateString(doc_id) : 0;
  auto codes__ = codes ? _fbb.CreateVector<uint8_t>(*codes) : 0;
  return lintdb::CreateInvertedDocument(
      _fbb,
      doc_id__,
      num_tokens,
      codes__);
}

inline const lintdb::InvertedDocument *GetInvertedDocument(const void *buf) {
  return ::flatbuffers::GetRoot<lintdb::InvertedDocument>(buf);
}

inline const lintdb::InvertedDocument *GetSizePrefixedInvertedDocument(const void *buf) {
  return ::flatbuffers::GetSizePrefixedRoot<lintdb::InvertedDocument>(buf);
}

inline bool VerifyInvertedDocumentBuffer(
    ::flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<lintdb::InvertedDocument>(nullptr);
}

inline bool VerifySizePrefixedInvertedDocumentBuffer(
    ::flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<lintdb::InvertedDocument>(nullptr);
}

inline void FinishInvertedDocumentBuffer(
    ::flatbuffers::FlatBufferBuilder &fbb,
    ::flatbuffers::Offset<lintdb::InvertedDocument> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedInvertedDocumentBuffer(
    ::flatbuffers::FlatBufferBuilder &fbb,
    ::flatbuffers::Offset<lintdb::InvertedDocument> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace lintdb

#endif  // FLATBUFFERS_GENERATED_SCHEMA_LINTDB_H_
