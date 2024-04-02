set(
    LINT_DB_TESTS
    index_test.cpp
    plaid_test.cpp
    rocksdb_test.cpp
    binarizer_test.cpp
)

add_executable(
  lintdb-tests
  ${LINT_DB_TESTS}
)


# we use BLAS and LAPACK
find_package(BLAS REQUIRED)
target_link_libraries(lintdb-tests PRIVATE ${BLAS_LIBRARIES})
find_package(OpenBLAS CONFIG REQUIRED)
target_link_libraries(lintdb-tests PRIVATE OpenBLAS::OpenBLAS)
# find_package(LAPACK REQUIRED)
# target_link_libraries(lintdb-tests PRIVATE ${LAPACK_LIBRARIES})

find_package(Microsoft.GSL CONFIG REQUIRED)
target_link_libraries(lintdb-tests PRIVATE Microsoft.GSL::GSL)

enable_testing()

find_package(GTest CONFIG REQUIRED)

include(FetchContent)
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)


# target_include_directories(lintdb-tests PUBLIC
# $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

target_link_libraries(lintdb-tests PRIVATE lintdb)

target_link_libraries(lintdb-tests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)


include(GoogleTest)
gtest_discover_tests(lintdb-tests)