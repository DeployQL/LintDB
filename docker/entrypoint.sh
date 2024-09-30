mkdir build
cd build

conan install .. --build=missing -o with_ut=True -o build_tests=True -o build_benchmarks=True -o build_python=True -s compiler.libcxx=libstdc++11 -s build_type=Release
conan build ..