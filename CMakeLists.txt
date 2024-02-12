  
cmake_minimum_required(VERSION 3.25)

# had to set the following env variables
# export C_INCLUDE_PATH=$(llvm-config --includedir)
# export CPLUS_INCLUDE_PATH=$(llvm-config --includedir)
# export LIBRARY_PATH=$(llvm-config --libdir)

# need libomp-dev installed. using clang 14. clang can be installed from the llvm website using their script.

include(CMakeFindDependencyMacro)

# Setup vcpkg script with CMake (note: should be placed before project() call)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(lintdb)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

add_subdirectory(lintdb)
# add_subdirectory(test)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lintdb PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(OpenBLAS CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE OpenBLAS::OpenBLAS)

find_package(LAPACK CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE LAPACK::LAPACK)
  
# this is heuristically generated, and may not be correct
find_package(faiss CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE faiss)

# this is heuristically generated, and may not be correct
find_package(RocksDB CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE RocksDB::rocksdb)

find_package(flatbuffers CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE flatbuffers::flatbuffers)

include(CTest)
add_subdirectory(tests)

# if(BUILD_TESTING)
# endif()