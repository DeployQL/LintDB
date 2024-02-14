  
cmake_minimum_required(VERSION 3.25)

# had to set the following env variables
# export C_INCLUDE_PATH=$(llvm-config --includedir)
# export CPLUS_INCLUDE_PATH=$(llvm-config --includedir)
# export LIBRARY_PATH=$(llvm-config --libdir)

# need libomp-dev installed. using clang 14. clang can be installed from the llvm website using their script.

include(CMakeFindDependencyMacro)

# Setup vcpkg script with CMake (note: should be placed before project() call)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tools/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(lintdb)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fopenmp -pthread")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(AF_COMPUTE_LIBRARY "FFTW/LAPACK/BLAS")
set(USE_CPU_MKL:BOOL OFF)
set(USE_OPENCL_MKL OFF)
set(AF_BUILD_CPU OFF)
set(AF_BUILD_CUDA OFF)
set(AF_BUILD_OPENCL OFF)
set(AF_BUILD_ONEAPI OFF)
set(AF_BUILD_EXAMPLES OFF)
set(AF_BUILD_TESTS OFF)

add_subdirectory(lintdb)
# add_subdirectory(test)

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

find_package(ArrayFire CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE ArrayFire::af ArrayFire::afcpu)

find_package(glog CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE glog::glog)

include(CTest)
add_subdirectory(tests)

# if(BUILD_TESTING)
# endif()