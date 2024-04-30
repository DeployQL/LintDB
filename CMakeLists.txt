  
cmake_minimum_required(VERSION 3.25)

include(CMakeFindDependencyMacro)

# allow faiss to build on m1 mac even though it's listed as unsupported.
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -D_LIBCPP_DISABLE_AVAILABILITY")
if(LINUX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
# the below is caused by github actions failing to build flatbuffers.
# therefore, we set this value so that we use a higher sdk version to build it.
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13)
# https://conda-forge.org/docs/maintainer/knowledge_base/#newer-c-features-with-old-sdk
# if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
# endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

option(ENABLE_PYTHON "Build Python extension." ON)
option(ENABLE_BENCHMARKS "Build benchmarks." OFF)

add_subdirectory(lintdb)

if (ENABLE_PYTHON)
  add_subdirectory(lintdb/python)
endif()

include(CTest)
if(BUILD_TESTING)
add_subdirectory(tests)
endif()

if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()