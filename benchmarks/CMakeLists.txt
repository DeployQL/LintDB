add_executable(bench_lintdb EXCLUDE_FROM_ALL bench_lintdb.cpp)
target_link_libraries(bench_lintdb PRIVATE lintdb)

find_package(benchmark CONFIG REQUIRED)
target_link_libraries(bench_lintdb PRIVATE benchmark::benchmark benchmark::benchmark_main)

find_package(MKL)
if (MKL_FOUND)
  # target_link_libraries(faiss PRIVATE "-Wl,--start-group" ${MKL_LIBRARIES} "-Wl,--end-group")
  # target_link_libraries(lintdb PRIVATE MKL::MKL)
  target_link_libraries(bench_lintdb PRIVATE ${MKL_LIBRARIES})
else()
find_package(OpenBLAS CONFIG REQUIRED)
target_link_libraries(bench_lintdb PRIVATE OpenBLAS::OpenBLAS)

find_package(LAPACK CONFIG REQUIRED)
target_link_libraries(bench_lintdb PRIVATE LAPACK::LAPACK)
endif()