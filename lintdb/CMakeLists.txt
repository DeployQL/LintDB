
set(
    LINT_DB_SRC
    Collection.cpp
    index.cpp
    quantizers/Binarizer.cpp
    Encoder.cpp
    quantizers/ProductEncoder.cpp
    quantizers/io.cpp
    util.cpp
    index_builder/EmbeddingModel.cpp
    index_builder/Tokenizer.cpp
    invlists/RocksdbForwardIndex.cpp
    invlists/keys.cpp
    invlists/EncodedDocument.cpp
    retrievers/PlaidRetriever.cpp
    retrievers/plaid.cpp
    retrievers/EMVBRetriever.cpp
    schema/util.cpp
    invlists/RocksdbInvertedListV2.cpp
    invlists/RocksdbInvertedList.cpp
    retrievers/XTRRetriever.cpp
    quantizers/InvertedListScanner.cpp
    quantizers/PQDistanceTables.cpp
)

set(
    LINT_DB_HEADERS
    index.h
    api.h
    assert.h
    exception.h
    Collection.h
    constants.h
    cf.h
    EmbeddingBlock.h
    Encoder.h
    env.h
    Passages.h
    quantizers/Binarizer.h
    quantizers/Quantizer.h
    quantizers/ProductEncoder.h
    quantizers/io.h
    SearchOptions.h
    SearchResult.h
    util.h
    index_builder/EmbeddingModel.h
    index_builder/Tokenizer.h
    invlists/InvertedList.h
    invlists/Iterator.h
    invlists/EncodedDocument.h
    invlists/RocksdbForwardIndex.h
    invlists/RocksdbInvertedListV2.h
    quantizers/PQDistanceTables.h
    quantizers/InvertedListScanner.h
    invlists/RocksdbInvertedList.h
    retrievers/XTRRetriever.h
    invlists/keys.h
    retrievers/Retriever.h
    retrievers/PlaidRetriever.h
    retrievers/plaid.h
    retrievers/EMVBRetriever.h
    retrievers/emvb_util.h
    retrievers/emvb.h
    retrievers/emvb_generic.h
    retrievers/emvb_avx.h
    schema/forward_index_generated.h
    schema/inverted_index_generated.h
    schema/mapping_generated.h
    schema/util.h
    utils/progress_bar.h
    version.h
)

set(LINT_DB_HEADERS ${LINT_DB_HEADERS} PARENT_SCOPE)

add_library(lintdb ${LINT_DB_SRC})

string(FIND "${CMAKE_CXX_FLAGS}" "FINTEGER" finteger_idx)
if (${finteger_idx} EQUAL -1)
  target_compile_definitions(lintdb PRIVATE FINTEGER=int)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lintdb PUBLIC OpenMP::OpenMP_CXX)
endif()

message(STATUS "using MKL root: $ENV{MKLROOT}")
find_package(MKL)
if (MKL_FOUND)
  message(STATUS "libraries: ${MKL_LIBRARIES}")
  target_link_libraries(lintdb PRIVATE ${MKL_LIBRARIES})
else()
find_package(OpenBLAS CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE OpenBLAS::OpenBLAS)

find_package(LAPACK CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE LAPACK::LAPACK)
endif()
  
# this is heuristically generated, and may not be correct
find_package(faiss CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE faiss)

# this is heuristically generated, and may not be correct
find_package(RocksDB CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE RocksDB::rocksdb)

find_package(flatbuffers CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE flatbuffers::flatbuffers)

find_package(glog CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE glog::glog)

# this is heuristically generated, and may not be correct
find_package(Microsoft.GSL CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE Microsoft.GSL::GSL)

find_package(jsoncpp CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE JsonCpp::JsonCpp)

find_package(Bitsery CONFIG REQUIRED)
target_link_libraries(lintdb PRIVATE Bitsery::bitsery)

# # this is heuristically generated, and may not be correct
# find_package(protobuf CONFIG REQUIRED)
# target_link_libraries(lintdb PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

# this is always dynamically linked! This places more responsbility on the user's environment!
find_library(ONNXRUNTIME_LIBRARY onnxruntime)
target_link_libraries(lintdb PRIVATE ${ONNXRUNTIME_LIBRARY})


set(TOKENZIER_CPP_PATH ${PROJECT_SOURCE_DIR}/lib/tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)
target_include_directories(lintdb PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_link_libraries(lintdb PRIVATE tokenizers_cpp)

if(MSVC)
  target_compile_options(lintdb PRIVATE /W4 /WX)
else()
  target_compile_options(lintdb PRIVATE -Wall -Wpedantic)
endif()

target_include_directories(lintdb PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(lintdb PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

foreach(header ${LINT_DB_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lintdb/${dir}
  )
endforeach()


install(TARGETS lintdb tokenizers_cpp tokenizers_c sentencepiece-static
  EXPORT lintdb-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/lintdb-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(${PROJECT_SOURCE_DIR}/cmake/lintdb-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/lintdb-config.cmake
  COPYONLY
)
install(FILES ${PROJECT_BINARY_DIR}/cmake/lintdb-config.cmake
  ${PROJECT_BINARY_DIR}/cmake/lintdb-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/lintdb
)

install(EXPORT lintdb-targets
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/lintdb
)
