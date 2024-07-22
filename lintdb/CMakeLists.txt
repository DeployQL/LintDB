
set(
    LINT_DB_SRC
    index.cpp
    quantizers/Binarizer.cpp
    quantizers/ProductEncoder.cpp
    quantizers/io.cpp
    util.cpp
    index_builder/EmbeddingModel.cpp
    index_builder/Tokenizer.cpp
    invlists/RocksdbForwardIndex.cpp
    invlists/EncodedDocument.cpp
        invlists/IndexWriter.cpp
#    retrievers/PlaidRetriever.cpp
#    retrievers/plaid.cpp
#    retrievers/EMVBRetriever.cpp
        invlists/RocksdbInvertedList.cpp
        invlists/ForwardIndexIterator.cpp
        invlists/IndexWriter.cpp
        invlists/InvertedIterator.cpp

#    retrievers/XTRRetriever.cpp
#    quantizers/InvertedListScanner.cpp
    quantizers/PQDistanceTables.cpp
    quantizers/impl/kmeans.cpp
    quantizers/CoarseQuantizer.cpp
        query/DocIterator.cpp
        query/Query.cpp
        query/QueryNode.cpp
        schema/DocEncoder.cpp
    schema/DocProcessor.cpp
    schema/Schema.cpp
        schema/FieldMapper.cpp
        query/QueryExecutor.cpp
        scoring/Scorer.cpp
        query/KnnNearestCentroids.cpp
        quantizers/IdentityQuantizer.cpp

        scoring/plaid.cpp
        query/decode.cpp
)

set(
    LINT_DB_HEADERS
    index.h
    api.h
    assert.h
    exception.h
    constants.h
    cf.h
        env.h
        quantizers/Binarizer.h
    quantizers/Quantizer.h
    quantizers/ProductEncoder.h
    quantizers/io.h
        query/DocIterator.h
        query/Query.h
        query/QueryNode.h
    SearchOptions.h
    SearchResult.h
    util.h
    index_builder/EmbeddingModel.h
    index_builder/Tokenizer.h
        invlists/IndexWriter.h
    invlists/InvertedList.h
    invlists/Iterator.h
    invlists/EncodedDocument.h
    invlists/RocksdbForwardIndex.h
        invlists/InvertedIterator.h
    quantizers/PQDistanceTables.h
#    quantizers/InvertedListScanner.h
    quantizers/impl/product_quantizer.h
    quantizers/CoarseQuantizer.h
    quantizers/impl/kmeans.h
        quantizers/IdentityQuantizer.h

        invlists/RocksdbInvertedList.h

#    retrievers/XTRRetriever.h
        invlists/ForwardIndexIterator.h
        invlists/IndexWriter.h
        invlists/PostingData.h
        invlists/ContextIterator.h
    retrievers/Retriever.h
#    retrievers/PlaidRetriever.h
#    retrievers/plaid.h
#    retrievers/EMVBRetriever.h
#    retrievers/emvb_util.h
#    retrievers/emvb.h
#    retrievers/emvb_generic.h
#    retrievers/emvb_avx.h
        schema/Schema.h
        schema/DocEncoder.h
      schema/DocProcessor.h
      schema/Document.h
      schema/DataTypes.h
        schema/FieldMapper.h
      schema/ProcessedData.h
        query/QueryContext.h
        query/QueryExecutor.h
        scoring/Scorer.h
        scoring/plaid.h
      query/KnnNearestCentroids.h
        invlists/KeyBuilder.h
        utils/endian.h
        query/decode.h
    utils/progress_bar.h
    version.h

)

set(LINT_DB_HEADERS ${LINT_DB_HEADERS} PARENT_SCOPE)

add_library(lintdb ${LINT_DB_SRC})

string(FIND "${CMAKE_CXX_FLAGS}" "FINTEGER" finteger_idx)
if (${finteger_idx} EQUAL -1)
  target_compile_definitions(lintdb PRIVATE FINTEGER=int)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "found openMP")
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

# this is always dynamically linked! This places more responsbility on the user's environment!
find_library(ONNXRUNTIME_LIBRARY onnxruntime)
target_link_libraries(lintdb PRIVATE ${ONNXRUNTIME_LIBRARY})


set(TOKENZIER_CPP_PATH ${PROJECT_SOURCE_DIR}/third_party/tokenizers-cpp)
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
