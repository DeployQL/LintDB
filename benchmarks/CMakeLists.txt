
message(STATUS "Building lintdb benchmark")

add_executable(bench_lintdb bench_lintdb.cpp)

target_link_libraries(bench_lintdb PRIVATE lintdb)

find_package(benchmark CONFIG REQUIRED)
target_link_libraries(bench_lintdb PRIVATE benchmark::benchmark benchmark::benchmark_main)

install(TARGETS bench_lintdb DESTINATION bin/benchmarks)