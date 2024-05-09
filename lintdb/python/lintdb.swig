/* -*- C -*-  (not really, but good for syntax highlighting) */

%module lintdb
%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdint.h>
#include <omp.h>
#include <map>
#include <utility>
#include <string>
#include <memory>
#include <cstdint>

#include "lintdb/SearchResult.h"
#include "lintdb/index.h"
#include "lintdb/SearchOptions.h"
#include "lintdb/RawPassage.h"
#include "lintdb/Collection.h"
#include "lintdb/exception.h"

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
%include std_string.i
%include std_pair.i
%include std_map.i
%include cpointer.i

%include <stdint.i>
// Apply enables the typemap for the given argument(s) in the function signature.
// without this, python was complaining about the idx_t.
%apply int64_t { idx_t };

%include <std_vector.i>

namespace std {
    
    %template(MetadataMap) std::map<std::string, std::string>;

    %template(FloatVector) vector<float>;
    %template(IdxVector) vector<idx_t>;
    // the below work to use append to our own vector, but not for add.
    %template(RawPassageVector) vector<lintdb::RawPassage>;
    %template(RawPassagePtrVector) vector<lintdb::RawPassage *>;
    %template(RawPassageConstPtrVector) vector<lintdb::RawPassage const *>;
    %template(SearchResultVector) vector<lintdb::SearchResult>;
    %template() vector<std::string>;
}


%init %{
    /* needed, else crash at runtime */
    import_array();
%}

%template() std::map<std::string, std::string>;
// %template() std::pair<swig::SwigPtr_PyObject, swig::SwigPtr_PyObject>;
// %template(pymap) std::map<swig::SwigPtr_PyObject, swig::SwigPtr_PyObject>;

// Apply a typemap for SearchResult map results.
%typemap(out) std::map lintdb::SearchResult::metadata  {
    PyObject* obj = PyDict_New();
    for (const auto& it : $1) {
        PyObject* key = PyUnicode_FromString(it->first.c_str());
        PyObject* value = PyUnicode_FromString(it->second.c_str());
        PyDict_SetItem(obj, key, value);
        Py_XDECREF(key);
        Py_XDECREF(value);
    }
    $result = SWIG_Python_AppendOutput($result, obj);
}

%typemap(out) std::map *lintdb::SearchResult::metadata  {
    PyObject* obj = PyDict_New();
    for (const auto& it : *$1) {
        PyObject* key = PyUnicode_FromString(it.first.c_str());
        PyObject* value = PyUnicode_FromString(it.second.c_str());
        PyDict_SetItem(obj, key, value);
        Py_XDECREF(key);
        Py_XDECREF(value);
    }
    $result = SWIG_Python_AppendOutput($result, obj);
}

// RawPassage constructor.
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* block, int num_tokens, int dim)};
// arguments have to be exact to the C++ function signature. e.g. embeddings, n, dim.
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* embeddings, int n, int dim)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* embeddings, const size_t num_tokens, const size_t dimensions)};
// apply it to search.
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float* data, const int n, const int dim)};
// apply to set_centroids
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* data, int n, int dim)};


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
    Py_BEGIN_ALLOW_THREADS
    try {
        $action
    } catch(LintDBException & e) {
        PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
     } catch(faiss::FaissException & e) {
        PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
    } catch(std::bad_alloc & ba) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_MemoryError, "std::bad_alloc");
        SWIG_fail;
    } catch(const std::exception& ex) {
        PyEval_RestoreThread(_save);
        std::string what = std::string("C++ exception ") + ex.what();
        PyErr_SetString(PyExc_RuntimeError, what.c_str());
        SWIG_fail;
    }
    Py_END_ALLOW_THREADS
}

// Define the C++ class
%include "lintdb/RawPassage.h"  
%include "lintdb/SearchOptions.h"
%include "lintdb/SearchResult.h"
%include "lintdb/index.h"
%include "lintdb/Collection.h"
