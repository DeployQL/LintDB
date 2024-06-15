#!/bin/sh

set -e

MKLROOT=_build/vcpkg_installed/x64-linux/lib/intel64 cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DENABLE_PYTHON=OFF \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DBLA_VENDOR=Intel10_64lp \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) lintdb

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _liblintdb_stage/