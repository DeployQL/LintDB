set(LINT_DB_TESTS
    util.h
    coarse_quantizer_test.cpp
    doc_iterator_test.cpp
    index_test.cpp
    mocks.h
    keys_test.cpp
    doc_encoder_test.cpp
    colbert_test.cpp
    plaid_test.cpp
    binarizer_test.cpp
    inverted_list_test.cpp
    doc_processor_test.cpp
    product_quantizer_test.cpp)

add_executable(lintdb_tests ${LINT_DB_TESTS})


find_package(Bitsery CONFIG REQUIRED)
find_package(RocksDB REQUIRED)
target_link_libraries(lintdb_tests PRIVATE Bitsery::bitsery)
target_link_libraries(lintdb_tests PRIVATE lintdb RocksDB::rocksdb)

target_compile_options(
        lintdb_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -mfma
        -mf16c
        -mavx2
        -msse4.2
        -mpopcnt
        -Wno-sign-compare
        -Wno-unused-variable
        -Wno-reorder
        -Wno-unused-local-typedefs
        -Wno-unused-function
        -Wno-strict-aliasing>)

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

target_link_libraries(lintdb_tests PRIVATE GTest::gtest GTest::gtest_main
                                           GTest::gmock GTest::gmock_main)

include(GoogleTest)
gtest_discover_tests(lintdb_tests
                     WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests")
