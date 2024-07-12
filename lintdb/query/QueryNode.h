#ifndef LINTDB_QUERYNODE_H
#define LINTDB_QUERYNODE_H

#include <limits>
#include <vector>
#include <string>
#include <memory>
#include "lintdb/invlists/EncodedDocument.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/invlists/RocksdbInvertedList.h"
#include "lintdb/query/DocIterator.h"
#include "lintdb/schema/DataTypes.h"
#include "lintdb/SearchOptions.h"

namespace lintdb {
    class QueryContext;

    enum class QueryNodeType {
        TERM,
        VECTOR,
        AND
    };

    class QueryNode {
    public:
        QueryNode() = delete;
        explicit QueryNode(QueryNodeType op): operator_(op) {};
        explicit QueryNode(QueryNodeType op, std::string& field, FieldValue& value): operator_(op), field(field), value(value) {};
        virtual std::unique_ptr<DocIterator> process(QueryContext& context, const SearchOptions& opts) = 0;

        virtual ~QueryNode() = default;

    protected:
        QueryNodeType operator_;
        std::string field;
        FieldValue value;
    };

    class TermQueryNode: public QueryNode {
    public:
        TermQueryNode() = delete;

        TermQueryNode(std::string field, FieldValue& value): QueryNode(QueryNodeType::TERM, field, value){};

        std::unique_ptr<DocIterator> process(QueryContext& context, const SearchOptions& opts) override;
    };

    class MultiQueryNode : public QueryNode {
    public:
        explicit MultiQueryNode(QueryNodeType op): QueryNode(op) {};

        inline void add_child(std::unique_ptr<QueryNode> child) {
            children_.push_back(std::move(child));
        }
    protected:
        std::vector<std::unique_ptr<QueryNode>> children_;

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
        VectorQueryNode(std::string field_id, FieldValue& value): QueryNode(QueryNodeType::VECTOR, field_id, value){};
        std::unique_ptr<DocIterator> process(QueryContext& context, const SearchOptions& opts) override;
    };

    class AndQueryNode : public MultiQueryNode {
    public:
        AndQueryNode() = delete;
        explicit AndQueryNode(std::vector<std::unique_ptr<QueryNode>> its): MultiQueryNode(QueryNodeType::AND), children_(std::move(its)) {};
        std::unique_ptr<DocIterator> process(QueryContext& context, const SearchOptions& opts) override;

    protected:
        std::vector<std::unique_ptr<QueryNode>> children_;
    };

}


#endif //LINTDB_QUERYNODE_H
