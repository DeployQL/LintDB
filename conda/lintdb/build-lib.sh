set -e


cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) lintdb

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _liblintdb_stage/