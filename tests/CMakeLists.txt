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
    product_quantizer_test.cpp
    query/DataFrameTest.cpp
    query/physical/PhysicalPlanTest.cpp
    query/physical/VectorSearchPhysicalPlanTest.cpp
    )

add_executable(lintdb_tests ${LINT_DB_TESTS})

# Add test data directory to include path
target_include_directories(lintdb_tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(Bitsery CONFIG REQUIRED)
find_package(RocksDB REQUIRED)
find_package(GTest CONFIG REQUIRED)

target_link_libraries(lintdb_tests PRIVATE 
    Bitsery::bitsery
    lintdb 
    RocksDB::rocksdb
    GTest::gtest 
    GTest::gtest_main
    GTest::gmock 
    GTest::gmock_main)

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

# Configure test discovery
include(GoogleTest)
gtest_discover_tests(lintdb_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        ENVIRONMENT "GTEST_COLOR=1"
        TIMEOUT 30
)
