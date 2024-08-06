#include "Query.h"

namespace lintdb {

Query::Query(std::unique_ptr<QueryNode> root) : root(std::move(root)) {}

} // namespace lintdb