

build:
	CC=clang CXX=clang++ cmake -S . -B build -DCMAKE_MAKE_PROGRAM=make -DCMAKE_BUILD_TYPE=Release

build-debug:
	CC=clang CXX=clang++ cmake -Wall -S . -B build -DCMAKE_MAKE_PROGRAM=make -DCMAKE_BUILD_TYPE=Debug

test: build
	export AF_PRINT_ERRORS=1
	export AF_TRACE=all
	export GLOG_v=2
	cd build && cmake -E env GLOG_v=10 GLOG_logtostderr=1 ctest --output-on-failure

prepare:
	sudo apt-get install xorg-dev gfortran libopenblas-dev liblapacke-dev libfftw3-dev clang cmake unzip
	sudo apt-get install libboost-all-dev