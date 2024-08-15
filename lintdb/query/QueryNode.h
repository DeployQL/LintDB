#ifndef LINTDB_QUERYNODE_H
#define LINTDB_QUERYNODE_H

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/invlists/RocksdbInvertedList.h"
#include "lintdb/query/DocIterator.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/SearchOptions.h"
#include <json/reader.h>
#include <json/writer.h>

namespace lintdb {
class QueryContext;

enum class QueryNodeType { TERM, VECTOR, AND };

class QueryNode {
   public:
    QueryNode() = delete;
    explicit QueryNode(QueryNodeType op) : operator_(op){};
    explicit QueryNode(QueryNodeType op, FieldValue& value)
            : operator_(op), value(value){};
    virtual std::unique_ptr<DocIterator> process(
            QueryContext& context,
            const SearchOptions& opts) = 0;

    virtual ~QueryNode() = default;

   protected:
    QueryNodeType operator_;
    FieldValue value;
};

class TermQueryNode : public QueryNode {
   public:
    TermQueryNode() = delete;

    TermQueryNode(FieldValue& value) : QueryNode(QueryNodeType::TERM, value){};

    std::unique_ptr<DocIterator> process(
            QueryContext& context,
            const SearchOptions& opts) override;
};

class MultiQueryNode : public QueryNode {
   public:
    explicit MultiQueryNode(QueryNodeType op) : QueryNode(op){};

    inline void add_child(std::unique_ptr<QueryNode> child) {
        children_.push_back(std::move(child));
    }

   protected:
    std::vector<std::unique_ptr<QueryNode>> children_ = {};
};

/**
 * VectorQueryNodes are query nodes for vector search.
 *
 * The returning DocIterator will search all IVF centroids based on this vector,
 * so it's behavior is similar to a multi-node queryNode using an OR operator.
 */
class VectorQueryNode : public QueryNode {
   public:
    VectorQueryNode() = delete;
    VectorQueryNode(FieldValue& value)
            : QueryNode(QueryNodeType::VECTOR, value){};
    std::unique_ptr<DocIterator> process(
            QueryContext& context,
            const SearchOptions& opts) override;
};

class AndQueryNode : public MultiQueryNode {
   public:
    AndQueryNode() = delete;
    explicit AndQueryNode(std::vector<std::unique_ptr<QueryNode>> its)
            : MultiQueryNode(QueryNodeType::AND) {
        for (auto& child : its) {
            children_.push_back(std::move(child));
        }
    };
    std::unique_ptr<DocIterator> process(
            QueryContext& context,
            const SearchOptions& opts) override;

   protected:
    std::vector<std::unique_ptr<QueryNode>> children_ = {};
};

} // namespace lintdb

#endif // LINTDB_QUERYNODE_H
