cmake_minimum_required(VERSION 3.25)

include(CMakeFindDependencyMacro)
include(GNUInstallDirs)
include(ExternalProject)
# allow faiss to build on m1 mac even though it's listed as unsupported.


file(READ "version.txt" version)

project(
  lintdb
  VERSION ${version}
  DESCRIPTION "A multi-vector database for late interaction retrieval"
  LANGUAGES CXX)
set(LINTDB_VERSION ${version})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17 -fPIC -Wall -pthread")

if (__x86_64__)
  set(CMAKE_CXX_FLAGS "-msse4.2 ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


if(MSVC OR LINUX)
#  set(BLA_VENDOR "Intel10_64lp")
#  set(BLA_VENDOR_THREADING "intel")
  set(BLA_VENDOR "OpenBLAS")
else()
    set(BLA_VENDOR "OpenBLAS")
endif()

set(CMAKE_CXX_STANDARD 17)
# the below is caused by github actions failing to build flatbuffers. therefore,
# we set this value so that we use a higher sdk version to build it.
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# OpenMP is getting pulled in through a version of blas. This needs to get figured out.
find_package(OpenMP REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/libs/libfaiss.cmake)
include_directories(third_party/faiss)

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(RocksDB REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(Bitsery CONFIG REQUIRED)
find_package(Drogon REQUIRED)

if(OPENMP_FOUND)
  message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS} -- ${OpenMP_EXE_LINKER_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(third_party/tantivy)

include_directories(lintdb)

# Collect source files
file(GLOB LINT_DB_SRC
        "lintdb/*.cpp"
        "lintdb/quantizers/*.cpp"
        "lintdb/quantizers/impl/*.cpp"
        "lintdb/invlists/*.cpp"
        "lintdb/query/*.cpp"
        "lintdb/schema/*.cpp"
        "lintdb/scoring/*.cpp"
)

# Collect header files
file(GLOB LINT_DB_HEADERS
        "lintdb/*.h"
        "lintdb/quantizers/*.h"
        "lintdb/quantizers/impl/*.h"
        "lintdb/invlists/*.h"
        "lintdb/query/*.h"
        "lintdb/schema/*.h"
        "lintdb/scoring/*.h"
        "lintdb/utils/*.h"
)

add_library(lintdb ${LINT_DB_SRC})

target_compile_options(
        lintdb
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -mfma
        -mf16c
        -mavx2
        -msse4.2
        -mpopcnt
        -Wno-sign-compare
        -Wno-unused-variable
        -Wno-reorder
        -Wno-unused-local-typedefs
        -Wno-unused-function
        -Wno-strict-aliasing>)

target_link_libraries(
        lintdb PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
)

target_link_libraries(lintdb
        PUBLIC
        faiss
        JsonCpp::JsonCpp
        Bitsery::bitsery
        glog::glog
        Microsoft.GSL::GSL
        RocksDB::rocksdb
)

target_include_directories(lintdb
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)



option(BUILD_PYTHON "Build Python extension." OFF)
option(BUILD_BENCHMARKS "Build benchmarks." ON)
option(BUILD_TESTS "Build tests." OFF)
option(BUILD_SERVER "Build the server." OFF)


if(BUILD_PYTHON)
  message(STATUS "building python extension")
  add_subdirectory(lintdb/python)
endif()

IF(BUILD_SERVER)
  add_subdirectory(lintdb/server)
endif()

include(CTest)
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

install(TARGETS lintdb
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
)
foreach(header ${LINT_DB_HEADERS})
  get_filename_component(dir ${header} DIRECTORY)
  install(FILES ${header} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lintdb/${dir})
endforeach()