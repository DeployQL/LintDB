ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

build-release:
	CC=clang-18 CXX=clang++-18 MKLROOT=${ROOT_DIR}/builds/release/vcpkg_installed/x64-linux/lib/intel64 cmake --preset release
	CC=clang-18 CXX=clang++-18 MKLROOT=${ROOT_DIR}/builds/release/vcpkg_installed/x64-linux/lib/intel64 cmake --build -j12 --preset release

build-debug:
	MKLROOT=${ROOT_DIR}/builds/debug/vcpkg_installed/x64-linux/lib/intel64 cmake \
	--preset debug \
	-DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
	-DOpenMP_CXX_LIB_NAMES=libiomp5 \
	-DOpenMP_libiomp5_LIBRARY=${ROOT_DIR}/builds/debug/vcpkg_installed/x64-linux/lib/intel64/libiomp5.so \
	.

	cmake --build --preset debug -j12

build-python:
	MKLROOT=${ROOT_DIR}/builds/python/vcpkg_installed/x64-linux/lib/intel64 cmake \
	--preset python \
	-DCMAKE_CXX_COMPILER=clang++-18 \
	-DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
	-DOpenMP_CXX_LIB_NAMES=libiomp5 \
	-DOpenMP_libiomp5_LIBRARY=${ROOT_DIR}/builds/python/vcpkg_installed/x64-linux/lib/intel64/libiomp5.so \
	.

	cmake --build --preset python -j12
	cd builds/python/lintdb/python && python setup.py build

test:
	cd builds/debug && cmake -E env GLOG_v=5 GLOG_logtostderr=1 ctest --output-on-failure

test-python: build-python
# had to fix up conda to make this work--
# conda install -c conda-forge gcc=12.1.0
# https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin
	GLOG_v=100 PYTHONPATH="builds/python/lintdb/python/build/lib" pytest -v  tests/test_*.py

run-python: build-python
	PYTHONPATH="build/lintdb/python/build/lib" python tests/runner.py

debug-test:
	lldb build/tests/lintdb-tests  

prepare:
	sudo apt-get install xorg-dev gfortran libopenblas-dev liblapacke-dev libfftw3-dev clang cmake unzip
	sudo apt-get install libboost-all-dev

format:
	find ./lintdb -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i

valgrind:
	valgrind -s --trace-children=yes --track-origins=yes --keep-stacktraces=alloc-and-free --suppressions=debug/valgrind-python.supp env PYTHONPATH="_build_python_/lintdb/python/build/lib/lintdb" python benchmarks/bench_lintdb.py --index-path=experiments/py_index_bench_test-collection-xtr

callgrind:
	OMP_MAX_ACTIVE_LEVELS=2 OMP_THREAD_LIMIT=6 OMP_NUM_THREADS=6 PYTHONPATH="_build_python_/lintdb/python/build/lib/lintdb" valgrind --tool=callgrind --suppressions=debug/valgrind-python.supp --instr-atstart=yes --dump-instr=yes --collect-jumps=yes python ./benchmarks/bench_lintdb.py single-search
	
callgrind-colbert: build-conda
	PYTHONPATH="_build_python_/lintdb/python/build/lib/lintdb" valgrind --tool=callgrind --suppressions=debug/valgrind-python.supp --instr-atstart=no --dump-instr=yes --collect-jumps=yes python ./benchmarks/run_colbert.py

py-docs:
	rm -rf docs/build
	sphinx-apidoc -o docs/source/ ./builds/python/lintdb/python/lintdb
	cd docs && make html
	cp icon.svg docs/build/html/icon.svg

debug-conda:
	conda debug lintdb --python 3.10 --output-id 'lintdb-*-py*' 

build-conda:
	CC=clang-18 CXX=clang++-18 MKLROOT=${ROOT_DIR}/_build_python_/vcpkg_installed/x64-linux/lib/intel64 cmake -B _build_python_${PY_VER} \
	-DBUILD_SHARED_LIBS=ON \
	-DENABLE_PYTHON=ON \
	-DBUILD_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DBLA_VENDOR=Intel10_64lp \
	-DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
	-DOpenMP_CXX_LIB_NAMES=libiomp5 \
	-DOpenMP_libiomp5_LIBRARY=${ROOT_DIR}/_build_python_/vcpkg_installed/x64-linux/lib/intel64/libiomp5.so \
	.

	cmake --build _build_python_${PY_VER} --target lintdb -j12
	cd _build_python_/lintdb/python && python setup.py build

build-benchmarks:
	MKLROOT=${ROOT_DIR}/build_benchmarks/vcpkg_installed/x64-linux/lib/intel64 cmake -B build_benchmarks \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_BENCHMARKS=ON \
	  -DENABLE_PYTHON=OFF \
	  -DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
      -DOpenMP_CXX_LIB_NAMES=libiomp5 \
	  -DOpenMP_libiomp5_LIBRARY=${ROOT_DIR}/build_benchmarks/vcpkg_installed/x64-linux/lib/intel64/libiomp5.so \
	  -DBLA_VENDOR=Intel10_64lp \
	  .
	CC=gcc CXX=g++ CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ cmake --build build_benchmarks --target=bench_lintdb -j12

run-perf:
# make sure your system allows perf to run. ex: sudo sysctl -w kernel.perf_event_paranoid=1 
	OMP_MAX_ACTIVE_LEVELS=2 OMP_THREAD_LIMIT=12 OMP_NUM_THREADS=6 PYTHONPATH="builds/python/lintdb/python/build/lib/lintdb" perf record -g -- /home/matt/miniconda3/envs/lintdb-benchmark/bin/python -X perf benchmarks/run_lintdb.py
	perf script | ./debug/stackcollapse-perf.pl | ./debug/flamegraph.pl > perf.data.svg 