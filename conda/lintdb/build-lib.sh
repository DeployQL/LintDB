#!/bin/sh

set -e

MKLROOT=_build/vcpkg_installed/x64-linux/lib/intel64 cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DENABLE_PYTHON=OFF \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DBLA_VENDOR=Intel10_64lp \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
      -DOpenMP_CXX_LIB_NAMES=libiomp5 \
      -DOpenMP_libiomp5_LIBRARY=${ROOT_DIR}/_build_python_/vcpkg_installed/x64-linux/lib/intel64/libiomp5.so \
      .

MKLROOT=_build/vcpkg_installed/x64-linux/lib/intel64 make -C _build -j$(nproc) lintdb

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _liblintdb_stage/