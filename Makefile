

build:
	CC=clang CXX=clang++ cmake -S . -B build -DCMAKE_MAKE_PROGRAM=make -DCMAKE_BUILD_TYPE=Release

build-debug:
	CC=clang CXX=clang++ cmake -Wall -S . -B build -DCMAKE_MAKE_PROGRAM=make -DCMAKE_BUILD_TYPE=Debug

test: build
	export AF_PRINT_ERRORS=1
	export AF_TRACE=all
	cd build && cmake -E env GLOG_v=10 GLOG_logtostderr=1 ctest --output-on-failure

test-python:
# had to fix up conda to make this work--
# conda install -c conda-forge gcc=12.1.0
# https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin
	cmake --build build -j12
	cd build/lintdb/python && python setup.py build
	PYTHONPATH="build/lintdb/python/build/lib" pytest tests/test_*.py

run-python:
	cmake --build build -j12
	cd build/lintdb/python && python setup.py build
	PYTHONPATH="build/lintdb/python/build/lib" python tests/runner.py

debug-test:
	lldb build/tests/lintdb-tests  

prepare:
	sudo apt-get install xorg-dev gfortran libopenblas-dev liblapacke-dev libfftw3-dev clang cmake unzip
	sudo apt-get install libboost-all-dev