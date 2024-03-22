

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(pylintdb
    LANGUAGES CXX
)

set(PYTHON_INCLUDE)

if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
endif()

set(CMAKE_CXX_STANDARD 17)
find_package(SWIG REQUIRED COMPONENTS python)
find_package(Python 3.10 REQUIRED COMPONENTS NumPy Development.Module)
find_package(RocksDB CONFIG REQUIRED)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

set(SRC lintdb.swig)
set_source_files_properties(${SRC} PROPERTIES CPLUSPLUS ON SWIG_FLAGS "-doxygen")
swig_add_library(pylintdb
    LANGUAGE python
    SOURCES ${SRC}
)

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
set_source_files_properties(${SRC} PROPERTIES
    SWIG_FLAGS -DSWIGWORDSIZE64
)
endif()
if(WIN32)
set_source_files_properties(${SRC} PROPERTIES
    SWIG_FLAGS -DSWIGWIN
)
endif()

set_property(TARGET pylintdb PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
# set_property(TARGET pylintdb PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

IF(NOT WIND32)
    set_target_properties(pylintdb PROPERTIES SUFFIX .so)
else()
    target_compile_options(pylintdb PRIVATE /bigobj)
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(pylintdb PRIVATE 
    lintdb
    RocksDB::rocksdb
    Python::Module
    Python::NumPy
    OpenMP::OpenMP_CXX
)

target_include_directories(pylintdb PRIVATE ${PROJECT_SOURCE_DIR}/../..)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)