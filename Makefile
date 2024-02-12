

build:
	cmake --build build

test: build
	cd build && ctest --output-on-failure