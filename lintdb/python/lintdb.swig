/* -*- C -*-  (not really, but good for syntax highlighting) */

%module("threads"=1) lintdb
%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "lintdb/index.h"
#include "lintdb/RawPassage.h"
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/exception.h"
#include <stdint.h>
#include <omp.h>
#include <faiss/impl/FaissException.h>
#include <numpy/arrayobject.h>

using namespace lintdb;
%}

// these typedefs are only available to SWIG, not to the generated code.
typedef unsigned long size_t;
typedef int64_t idx_t;

%include "std_unique_ptr.i"
%include "numpy.i"
%include <typemaps.i>
%include <std_string.i>
%include cpointer.i

// this uses code that makes this specific to 64bit arch. 
// We should be able to define this in cmake, but it's not working for a reason I don't understand.
// #define SWIGWORDSIZE64
%include <stdint.i>
// Apply enables the typemap for the given argument(s) in the function signature.
// without this, python was complaining about the idx_t.
%apply int64_t { idx_t };

%include <std_vector.i>
// the below typedef let's us compile, but python will segfault. perhaps this isn't aproblem
// with SWIG, but we never enter the function in c++.
// typedef lintdb::RawPassage RawPassage;

namespace std {
    %template(FloatVector) vector<float>;
    %template(IdxVector) vector<idx_t>;
    // the below work to use append to our own vector, but not for add.
    %template(RawPassageVector) vector<lintdb::RawPassage>;
    %template(RawPassagePtrVector) vector<lintdb::RawPassage *>;
    %template(RawPassageConstPtrVector) vector<lintdb::RawPassage const *>;
    %template(SearchResultVector) vector<lintdb::SearchResult>;
}
// these do not work fo rappending.
// %template(RawPassageVector) std::vector<RawPassage>;

%init %{
    /* needed, else crash at runtime */
    import_array();
%}

// RawPassage constructor.
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* block, int num_tokens, int dim)};
// arguments have to be exact to the C++ function signature. e.g. embeddings, n, dim.
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* embeddings, int n, int dim)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* embeddings, const size_t num_tokens, const size_t dimensions)};
// apply it to search.
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* data, const int n, const int dim)};
// apply to set_centroids
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* data, int n, int dim)};


// %nodefaultctor IndexIVF;

%ignore IndexIVF(std::string path, Configuration& config);
%ignore IndexIVF::IndexIVF(const IndexIVF&, const std::string);
%ignore IndexIVF(const IndexIVF& other, const std::string path);
// below works to completely ignore the search methods overall.
// %ignore search;
%ignore search(
        const uint64_t tenant,
        const EmbeddingBlock& block,
        const size_t n_probe,
        const size_t k,
        const SearchOptions& opts=SearchOptions()) const;
%ignore train(float*, size_t, size_t);
%ignore train(size_t n,std::vector<float>& embeddings);
%ignore RawPassage::RawPassage();
%ignore IndexIVF::invlists;
%ignore IndexIVF::decode_vectors;
%ignore IndexIVF::get_pids;


%ignore EmbeddingBlock::EmbeddingBlock(const float* embeddings, size_t num_tokens, size_t dimensions);

// Python-specific: release GIL by default for all functions
%exception {
    // Py_BEGIN_ALLOW_THREADS
    try {
        $action
    } catch(LintDBException & e) {
        // PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
     } catch(faiss::FaissException & e) {
        // PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
    } catch(std::bad_alloc & ba) {
        // PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_MemoryError, "std::bad_alloc");
        SWIG_fail;
    } catch(const std::exception& ex) {
        // PyEval_RestoreThread(_save);
        std::string what = std::string("C++ exception ") + ex.what();
        PyErr_SetString(PyExc_RuntimeError, what.c_str());
        SWIG_fail;
    }
    // Py_END_ALLOW_THREADS
}

// Define the C++ class
%include "lintdb/EmbeddingBlock.h"
%include "lintdb/RawPassage.h"  
%include "lintdb/index.h"
