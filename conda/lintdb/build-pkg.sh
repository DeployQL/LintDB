set -e

      # -DPython_EXECUTABLE=$PYTHON \
      # -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
      # -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \

MKLROOT=_build_python_${PY_VER}/vcpkg_installed/x64-linux/lib/intel64 cmake -B _build_python_${PY_VER} \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -Dlintdb_ROOT=_liblintdb_stage/ \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
      .

MKLROOT=_build_python_${PY_VER}/vcpkg_installed/x64-linux/lib/intel64 make -C _build_python_${PY_VER} -j$(nproc) pylintdb

# Build actual python module.
cd _build_python_${PY_VER}/lintdb/python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX