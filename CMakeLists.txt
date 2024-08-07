cmake_minimum_required(VERSION 3.25)

include(CMakeFindDependencyMacro)

# allow faiss to build on m1 mac even though it's listed as unsupported.
set(VCPKG_INSTALL_OPTIONS "--allow-unsupported")

# Setup vcpkg script with CMake (note: should be placed before project() call)
set(CMAKE_TOOLCHAIN_FILE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
    CACHE STRING "Vcpkg toolchain file")

file(READ "version.txt" version)

project(
  lintdb
  VERSION ${version}
  DESCRIPTION "A multi-vector database for late interaction retrieval"
  LANGUAGES CXX)
set(LINTDB_VERSION ${version})

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -O3 -D_LIBCPP_DISABLE_AVAILABILITY"
)

if(MSVC OR LINUX)
  set(BLA_VENDOR "Intel10_64lp")
else()
    set(BLA_VENDOR "OpenBLAS")
endif()

# the below is caused by github actions failing to build flatbuffers. therefore,
# we set this value so that we use a higher sdk version to build it.
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# https://conda-forge.org/docs/maintainer/knowledge_base/#newer-c-features-with-old-sdk
# if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") set(CMAKE_CXX_FLAGS
# "${CMAKE_CXX_FLAGS} ") endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if(SKBUILD)
  message("Building with scikit-build")
  set(ENV{MKLROOT}
      "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-linux/lib/intel64")
  set(OpenMP_libiomp5_LIBRARY
      "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-linux/lib/intel64/libiomp5.so"
  )
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

option(ENABLE_PYTHON "Build Python extension." ON)
option(ENABLE_BENCHMARKS "Build benchmarks." OFF)

add_subdirectory(lintdb)

if(ENABLE_PYTHON)
  add_subdirectory(lintdb/python)
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
