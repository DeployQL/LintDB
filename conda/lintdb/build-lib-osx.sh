set -e


cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
      -DOpenMP_CXX_LIB_NAMES=libiomp5 \
      -DOpenMP_libiomp5_LIBRARY=$PREFIX/lib/libiomp5.dylib \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DPython_EXECUTABLE=$PYTHON \
      -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
      -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) lintdb

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _liblintdb_stage/