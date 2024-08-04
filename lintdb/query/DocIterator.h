#pragma once

#include <memory>
#include <vector>
#include "DocValue.h"
#include "lintdb/api.h"
#include "lintdb/invlists/Iterator.h"
#include "lintdb/schema/DataTypes.h"

namespace lintdb {
/**
 * DocIterator helps us iterate over documents in the index.
 *
 * This encapsulates the logic of and/or and aligning index iterators to the
 * same document.
 */
class DocIterator {
   public:
    virtual void advance() = 0;
    virtual bool is_valid() = 0;

    virtual idx_t doc_id() const = 0;
    virtual std::vector<DocValue> fields() const = 0;

    virtual ~DocIterator() = default;
};

class TermIterator : public DocIterator {
   private:
    std::unique_ptr<Iterator> it_;

   public:
    TermIterator() = default;
    explicit TermIterator(
            std::unique_ptr<Iterator> it,
            DataType type,
            bool ignore_value = false);
    void advance() override;
    bool is_valid() override;

    idx_t doc_id() const override;
    std::vector<DocValue> fields() const override;

   private:
    bool ignore_value;
    DataType type;
};

class ANNIterator : public DocIterator {
   public:
    explicit ANNIterator(std::vector<std::unique_ptr<DocIterator>> its);
    void advance() override;
    bool is_valid() override;

    idx_t doc_id() const override;
    std::vector<DocValue> fields() const override;

   private:
    std::vector<DocValue> fields_;
    std::vector<std::unique_ptr<DocIterator>> its_;
    idx_t last_doc_id_;
    void heapify(size_t idx);
};

class AndIterator : public DocIterator {
   private:
    std::vector<std::unique_ptr<DocIterator>> its_;
    idx_t current_doc_id_;
    bool is_valid_;

    void synchronize();

   public:
    AndIterator(std::vector<std::unique_ptr<DocIterator>> iterators);
    void advance() override;
    bool is_valid() override;
    idx_t doc_id() const override;
    std::vector<DocValue> fields() const override;
};

} // namespace lintdb
