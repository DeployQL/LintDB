#ifndef LINTDB_ENV_H
#define LINTDB_ENV_H

namespace lintdb {
// environment variables we use to set the number of threads.
const char* ONNX_INTER_THREADS = "LINTDB_INTER_NUM_THREADS";
const char* ONNX_INTRA_THREADS = "LINTDB_INTRA_NUM_THREADS";
}

#endif // LINTDB_ENV_H
