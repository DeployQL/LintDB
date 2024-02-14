#ifndef LINTDB_INDEXSAVER_H
#define LINTDB_INDEXSAVER_H

#include <string>
#include "lintdb/index.h"

namespace lintdb {
    IndexIVF& load_index(const std::string& path);
}

#endif