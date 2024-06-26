

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(pylintdb
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
find_package(SWIG REQUIRED COMPONENTS python)
find_package(Python REQUIRED COMPONENTS NumPy Development.Module)
include(${SWIG_USE_FILE})

include(UseSWIG)

# below is useful for debugging typemaps. It will log how each object resolves its typemap.
# set(CMAKE_SWIG_FLAGS "-debug-tmsearch")

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

set(SRC lintdb.swig)


if(UNIX AND NOT APPLE)
    set_source_files_properties(${SRC} PROPERTIES 
        SWIG_FLAGS -DSWIGWORDSIZE64
    )
endif()
if(WIN32)
    set_source_files_properties(${SRC} PROPERTIES
        SWIG_FLAGS -DSWIGWIN
    )
endif()

set_source_files_properties(${SRC} PROPERTIES 
    CPLUSPLUS ON 
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
)
swig_add_library(pylintdb
    TYPE SHARED
    LANGUAGE python
    SOURCES ${SRC}
)

set_property(TARGET pylintdb PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
set_property(TARGET pylintdb PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)

IF(NOT WIN32)
    set_target_properties(pylintdb PROPERTIES SUFFIX .so)
else()
    target_compile_options(pylintdb PRIVATE /bigobj)
endif()

if(TARGET lintdb)
    set(SWIG_MODULE_pylintdb_EXTRA_DEPS)
    foreach(h ${LINT_DB_HEADERS})
        list(APPEND SWIG_MODULE_pylintdb_EXTRA_DEPS "${lintdb_SOURCE_DIR}/lintdb/${h}")
    endforeach()
else()
    find_package(lintdb REQUIRED)
endif()

find_package(Microsoft.GSL CONFIG REQUIRED)
target_link_libraries(pylintdb PRIVATE Microsoft.GSL::GSL)

find_package(OpenMP REQUIRED)

target_link_libraries(pylintdb PRIVATE 
    lintdb
    Python::Module
    Python::NumPy
    OpenMP::OpenMP_CXX
)

target_include_directories(pylintdb PRIVATE ${PROJECT_SOURCE_DIR}/../..)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)

# copy our version file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.txt ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
