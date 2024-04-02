set -e


cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DPython_EXECUTABLE=$PYTHON \
      -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
      -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      -DBLA_VENDOR=OpenBLAS \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) lintdb

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _liblintdb_stage/