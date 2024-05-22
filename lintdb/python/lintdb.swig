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
#include "lintdb/EmbeddingBlock.h"
#include "lintdb/Passages.h"
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

    %template() vector<float>;
    %template() vector<idx_t>;
    %template() vector<lintdb::SearchResult>;
    %template() vector<lintdb::EmbeddingPassage>;
    %template() vector<std::string>;
    %template() vector<lintdb::TextPassage>;
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

%typemap(in) lintdb::TextPassage* {
    if (!PyDict_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    PyObject* py_text = PyDict_GetItemString($input, "text");
    PyObject* py_id = PyDict_GetItemString($input, "id");
    PyObject* py_metadata = PyDict_GetItemString($input, "metadata");

    if (!py_text || !PyUnicode_Check(py_text) ||
        !py_id || !PyLong_Check(py_id) ||
        !py_metadata || !PyDict_Check(py_metadata)) {
        PyErr_SetString(PyExc_TypeError, "Dictionary keys are missing or of incorrect type");
        return NULL;
    }

    std::string text = PyUnicode_AsUTF8(py_text);
    int64_t id = PyLong_AsLongLong(py_id);
    std::map<std::string, std::string> metadata;

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(py_metadata, &pos, &key, &value)) {
        if (!PyUnicode_Check(key) || !PyUnicode_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "Metadata keys and values must be strings");
            return NULL;
        }
        metadata[PyUnicode_AsUTF8(key)] = PyUnicode_AsUTF8(value);
    }

    $1 = new TextPassage(text, id, metadata);
}

%typemap(freearg) lintdb::TextPassage* {
    delete $1;
}

%typemap(in) lintdb::TextPassage {
    if (!PyDict_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
        return NULL;
    }

    PyObject* py_text = PyDict_GetItemString($input, "text");
    PyObject* py_id = PyDict_GetItemString($input, "id");
    PyObject* py_metadata = PyDict_GetItemString($input, "metadata");

    if (!py_text || !PyUnicode_Check(py_text) ||
        !py_id || !PyLong_Check(py_id) ||
        !py_metadata || !PyDict_Check(py_metadata)) {
        PyErr_SetString(PyExc_TypeError, "Dictionary keys are missing or of incorrect type");
        return NULL;
    }

    std::string text = PyUnicode_AsUTF8(py_text);
    int64_t id = PyLong_AsLongLong(py_id);
    std::map<std::string, std::string> metadata;

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(py_metadata, &pos, &key, &value)) {
        if (!PyUnicode_Check(key) || !PyUnicode_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "Metadata keys and values must be strings");
            return NULL;
        }
        metadata[PyUnicode_AsUTF8(key)] = PyUnicode_AsUTF8(value);
    }

    $1 = TextPassage(text, id, metadata);
}

%typemap(in) std::vector<lintdb::TextPassage> (std::vector<lintdb::TextPassage> vec) {
    if (!PyList_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list");
        return NULL;
    }

    Py_ssize_t size = PyList_Size($input);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem($input, i);
        if (!PyDict_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "List items must be dictionaries");
            return NULL;
        }

        PyObject* py_text = PyDict_GetItemString(item, "text");
        PyObject* py_id = PyDict_GetItemString(item, "id");
        PyObject* py_metadata = PyDict_GetItemString(item, "metadata");

        if (!py_text || !PyUnicode_Check(py_text) ||
            !py_id || !PyLong_Check(py_id) ||
            !py_metadata || !PyDict_Check(py_metadata)) {
            PyErr_SetString(PyExc_TypeError, "Dictionary keys are missing or of incorrect type");
            return NULL;
        }

        std::string text = PyUnicode_AsUTF8(py_text);
        int64_t id = PyLong_AsLongLong(py_id);
        std::map<std::string, std::string> metadata;

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(py_metadata, &pos, &key, &value)) {
            if (!PyUnicode_Check(key) || !PyUnicode_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Metadata keys and values must be strings");
                return NULL;
            }
            metadata[PyUnicode_AsUTF8(key)] = PyUnicode_AsUTF8(value);
        }

        vec.emplace_back(text, id, metadata);
    }
    $1 = vec;
}


// EmbeddingPassage constructor.
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
%ignore IndexIVF::invlists;
%ignore IndexIVF::decode_vectors;
%ignore IndexIVF::get_pids;
%ignore IndexIVF::search(
        const uint64_t tenant,
        const EmbeddingBlock& block,
        const size_t k,
        const SearchOptions& opts = SearchOptions()) const;
%ignore IndexIVF::search(
            const uint64_t tenant,
            const float* data,
            const int n,
            const int dim,
            const size_t n_probe,
            const size_t k,
            const SearchOptions& opts = SearchOptions()) const;
%ignore IndexIVF::search(
            const uint64_t tenant,
            const EmbeddingBlock& block,
            const size_t n_probe,
            const size_t k,
            const SearchOptions& opts = SearchOptions()) const;


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

%rename (RawPassage) lintdb::EmbeddingPassage;
// Define the C++ class
%include "lintdb/Passages.h"
%include "lintdb/SearchOptions.h"
%include "lintdb/SearchResult.h"
%include "lintdb/index.h"
%include "lintdb/Collection.h"
