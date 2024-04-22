

build-release:

# CC=clang CXX=clang++ cmake -S . -B build -DCMAKE_MAKE_PROGRAM=make -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON
	cmake --preset release
	cmake --build -j12 --preset release

build-debug:
# CC=clang CXX=clang++ cmake -Wall -S . -B build -DCMAKE_MAKE_PROGRAM=make -DCMAKE_BUILD_TYPE=Debug -DLLDB_EXPORT_ALL_SYMBOLS=ON -DBUILD_SHARED_LIBS=ON
	MKLROOT=/home/matt/deployql/LintDB/builds/debug/vcpkg_installed/x64-linux/lib/intel64 cmake --preset debug
	MKLROOT=/home/matt/deployql/LintDB/builds/debug/vcpkg_installed/x64-linux/lib/intel64 cmake --build -j12 --preset debug

build-python: 
	cmake --preset python
	cmake --build --preset python -j12
	cd builds/python/lintdb/python && python setup.py build

test:
	cd builds/debug && cmake -E env GLOG_v=100 GLOG_logtostderr=1 MKL_THREADING_LAYER=GNU ctest --output-on-failure

test-python: build-python
# had to fix up conda to make this work--
# conda install -c conda-forge gcc=12.1.0
# https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin
	GLOG_v=100 PYTHONPATH="build/lintdb/python/build/lib" pytest tests/test_*.py

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
# we need valgrind?-3.20 to process dwarf5
	valgrind -s --trace-children=yes --track-origins=yes --keep-stacktraces=alloc-and-free --suppressions=debug/valgrind-python.supp env PYTHONPATH="build/lintdb/python/build/lib" python tests/test_index.py

callgrind:
	PYTHONPATH="_build_python_/lintdb/python/lintdb" valgrind --tool=callgrind python ./benchmarks/run_lintdb.py
	
py-docs:
	rm -rf docs/build
	sphinx-apidoc -o docs/source/ ./build/lintdb/python/lintdb
	cd docs && make html

debug-conda:
	conda debug lintdb --python 3.10 --output-id 'lintdb-*-py*' 

build-conda:
# this command mimicks how conda builds the package. it also makes it easier to build and augment the pythonpath than the regular build-python command
	cmake -B _build_python_${PY_VER} \
      -DBUILD_SHARED_LIBS=ON \
	  -DENABLE_PYTHON=ON \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
	  -DBLA_VENDOR=Intel10_64ilp \
      .
	cmake --build _build_python_${PY_VER} --target pylintdb -j12
	cd _build_python_/lintdb/python && python setup.py build 

build-benchmarks:
	MKLROOT=/home/matt/deployql/LintDB/builds/debug/vcpkg_installed/x64-linux/lib/intel64 cmake -B build_benchmarks \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_BENCHMARKS=ON \
	  -DENABLE_PYTHON=OFF \
	  -DBLA_VENDOR=Intel10_64lp \
	  .
	MKLROOT=/home/matt/deployql/LintDB/builds/debug/vcpkg_installed/x64-linux/lib/intel64 cmake --build build_benchmarks --target=bench_lintdb -j12