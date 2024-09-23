ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

build-release:
	MKLROOT=${ROOT_DIR}/builds/release/vcpkg_installed/x64-linux/lib/intel64 cmake \
	--preset release \
	-DCMAKE_CXX_COMPILER=clang++-18 \
	.

	cmake --build --preset release -j12

build-debug:
	MKLROOT=${ROOT_DIR}/builds/debug/vcpkg_installed/x64-linux/lib/intel64 cmake \
	--preset debug \
	-DCMAKE_CXX_COMPILER=g++ \
	.

	cmake --build --preset debug -v -j12 --target lintdb_lib
	cmake --build --preset debug -v -j12 --target lintdb-tests


build-python:
	MKLROOT=${ROOT_DIR}/builds/python/vcpkg_installed/x64-linux/lib/intel64 cmake \
	--preset python \
	-DCMAKE_CXX_COMPILER=clang++-18 \
	.

	cmake --build --preset python -j12 --target lintdb_lib
	cmake --build --preset python -j12 --target core

build-server:
	MKLROOT=${ROOT_DIR}/builds/server/vcpkg_installed/x64-linux/lib/intel64 cmake \
	--preset server \
	-DCMAKE_CXX_COMPILER=clang++-18 \
	-DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
	-DOpenMP_CXX_LIB_NAMES=libiomp5 \
	-DOpenMP_libiomp5_LIBRARY=${ROOT_DIR}/builds/server/vcpkg_installed/x64-linux/lib/intel64/libiomp5.so \
	.

	cmake --build --preset server -j12

build-benchmarks:
	MKLROOT=${ROOT_DIR}/builds/benchmarks/vcpkg_installed/x64-linux/lib/intel64 cmake \
	--preset benchmarks \
	-DCMAKE_CXX_COMPILER=clang++-18 \
	.

	cmake --build --preset benchmarks -j12  --target lintdb_lib
	cmake --build --preset benchmarks -v -j12 --target bench_lintdb

test:
	cd builds/debug && cmake -E env GLOG_v=5 GLOG_logtostderr=1 ctest --output-on-failure

format:
	find ./lintdb -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i

valgrind:
	valgrind -s --trace-children=yes --track-origins=yes --keep-stacktraces=alloc-and-free --suppressions=debug/valgrind-python.supp env PYTHONPATH="_build_python_/lintdb/python/build/lib/lintdb" python benchmarks/bench_lintdb.py --index-path=experiments/py_index_bench_test-collection-xtr

callgrind:
	OMP_MAX_ACTIVE_LEVELS=2 OMP_THREAD_LIMIT=6 OMP_NUM_THREADS=6 valgrind --tool=callgrind --suppressions=debug/valgrind-python.supp --instr-atstart=yes --dump-instr=yes --collect-jumps=yes python ./benchmarks/bench_lintdb.py index
	python -m gprof2dot --format=callgrind --output=out.dot callgrind.out.*
	dot -Tsvg out.dot -o graph.svg

callgrind-py:
	OMP_MAX_ACTIVE_LEVELS=2 OMP_THREAD_LIMIT=6 OMP_NUM_THREADS=6 valgrind --tool=callgrind --suppressions=debug/valgrind-python.supp --instr-atstart=yes --dump-instr=yes --collect-jumps=yes python ./benchmarks/lotte/indexing_two.py eval --index-path=benchmarks/lintdb-lifestyle-40k --stop=40000
	python -m gprof2dot --format=callgrind --output=out.dot callgrind.out.*
	dot -Tsvg out.dot -o graph.svg

callgrind-cpp: build-benchmarks
	valgrind --tool=callgrind --instr-atstart=yes --dump-instr=yes --collect-jumps=yes ./builds/benchmarks/benchmarks/Release/bench_lintdb
	python -m gprof2dot -n0 -e0 --format=callgrind --output=out.dot --strip callgrind.out.*
	dot -Tsvg out.dot -o graph.svg

callgrind-colbert: build-conda
	PYTHONPATH="_build_python_/lintdb/python/build/lib/lintdb" valgrind --tool=callgrind --suppressions=debug/valgrind-python.supp --instr-atstart=no --dump-instr=yes --collect-jumps=yes python ./benchmarks/run_colbert.py

run-perf:
# make sure your system allows perf to run. ex: sudo sysctl -w kernel.perf_event_paranoid=1 
	OMP_MAX_ACTIVE_LEVELS=2 OMP_THREAD_LIMIT=12 OMP_NUM_THREADS=6 perf record -g -- ./builds/benchmarks/benchmarks/Release/bench_lintdb
	perf script | ./debug/stackcollapse-perf.pl | ./debug/flamegraph.pl > perf.data.svg

run-docs:
	mkdocs serve -a 0.0.0.0:8000