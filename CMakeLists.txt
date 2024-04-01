  
cmake_minimum_required(VERSION 3.25)

# had to set the following env variables
# export C_INCLUDE_PATH=$(llvm-config --includedir)
# export CPLUS_INCLUDE_PATH=$(llvm-config --includedir)
# export LIBRARY_PATH=$(llvm-config --libdir)

# need libomp-dev installed. using clang 14. clang can be installed from the llvm website using their script.

include(CMakeFindDependencyMacro)

set(VCPKG_INSTALL_OPTIONS "--allow-unsupported")

# Setup vcpkg script with CMake (note: should be placed before project() call)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tools/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(lintdb
  VERSION 0.0.1
  DESCRIPTION "A multi-vector database for late interaction retrieval"
  LANGUAGES CXX
)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fopenmp -pthread")

# the below is caused by github actions failing to build flatbuffers.
# therefore, we set this value so that we use a higher sdk version to build it.
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13)
# https://conda-forge.org/docs/maintainer/knowledge_base/#newer-c-features-with-old-sdk
# if(APPLE)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_DISABLE_AVAILABILITY")
# endif()

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

add_subdirectory(lintdb)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lintdb PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(OpenBLAS CONFIG REQUIRED)
target_link_libraries(lintdb PUBLIC OpenBLAS::OpenBLAS)

find_package(LAPACK CONFIG REQUIRED)
target_link_libraries(lintdb PUBLIC LAPACK::LAPACK)
  
# this is heuristically generated, and may not be correct
find_package(faiss CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE faiss)

# this is heuristically generated, and may not be correct
find_package(RocksDB CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE RocksDB::rocksdb)

find_package(flatbuffers CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE flatbuffers::flatbuffers)

find_package(glog CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE glog::glog)

# this is heuristically generated, and may not be correct
find_package(Microsoft.GSL CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE Microsoft.GSL::GSL)

find_package(jsoncpp CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE JsonCpp::JsonCpp)

include(CTest)
add_subdirectory(tests)

# if(BUILD_TESTING)
# endif()