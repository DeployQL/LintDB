set -e

cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
      -DOpenMP_CXX_LIB_NAMES=libiomp5 \
      -DOpenMP_libiomp5_LIBRARY=$PREFIX/lib/libiomp5.dylib \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) lintdb

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _liblintdb_stage/