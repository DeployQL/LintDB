#!/bin/sh

set -e

#       -DBUILD_SHARED_LIBS=ON \
# I would love to use BUILD_SHARED_LIBS=ON, but it seems to be broken when running in conda.
cmake -B _build \
      -DBUILD_TESTING=OFF \
      -DENABLE_PYTHON=OFF \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DPython_EXECUTABLE=$PYTHON \
      -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
      -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) lintdb

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _liblintdb_stage/