#pragma once

#include "QueryNode.h"

namespace lintdb {
struct Query {
   public:
    Query(std::unique_ptr<QueryNode> root);

    std::unique_ptr<QueryNode> root;
};

} // namespace lintdb
