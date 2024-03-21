set -e


cmake -B _build_python_${PY_VER} \
      -Dlintdb_ROOT=_liblintdb_stage/ \
      -DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
      -DOpenMP_CXX_LIB_NAMES=libiomp5 \
      -DOpenMP_libiomp5_LIBRARY=$PREFIX/lib/libiomp5.dylib \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
      -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
      -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      lintdb/python

make -C _build_python_${PY_VER} -j$(nproc) pylintdb

# Build actual python module.
cd _build_python_${PY_VER}/
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX