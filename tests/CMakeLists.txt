set(
    LINT_DB_TESTS
    util.h
    index_test.cpp
    plaid_test.cpp
    rocksdb_test.cpp
    binarizer_test.cpp
    encoder_test.cpp
    index_builder_test.cpp
    mocks.h
    xtr_test.cpp
    inverted_list_test.cpp
    encoded_doc_test.cpp
)

add_executable(
  lintdb-tests
  ${LINT_DB_TESTS}
)

configure_file(${PROJECT_SOURCE_DIR}/assets/colbert_tokenizer.json colbert_tokenizer.json COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/assets/model.onnx model.onnx COPYONLY)

target_link_libraries(lintdb-tests PRIVATE lintdb)

find_package(Bitsery CONFIG REQUIRED)
target_link_libraries(lintdb-tests PRIVATE Bitsery::bitsery)

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

target_link_libraries(lintdb-tests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

include(GoogleTest)
gtest_discover_tests(lintdb-tests)