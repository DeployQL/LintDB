set -e

      # -DPython_EXECUTABLE=$PYTHON \
      # -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
      # -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \

cmake -B _build_python_${PY_VER} \
      -Dlintdb_ROOT=_liblintdb_stage/ \
      -DCMAKE_BUILD_TYPE=Release \
      .

make -C _build_python_${PY_VER} -j$(nproc) pylintdb

# Build actual python module.
cd _build_python_${PY_VER}/lintdb/python
$PYTHON setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --single-version-externally-managed --record=record.txt --prefix=$PREFIX