add_executable(bench_lintdb EXCLUDE_FROM_ALL bench_lintdb.cpp)
target_link_libraries(bench_lintdb PRIVATE lintdb)

find_package(benchmark CONFIG REQUIRED)
target_link_libraries(bench_lintdb PRIVATE benchmark::benchmark benchmark::benchmark_main)