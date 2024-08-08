set(LINT_DB_SRC
    index.cpp
    quantizers/Binarizer.cpp
    quantizers/ProductEncoder.cpp
    quantizers/io.cpp
    util.cpp
    invlists/RocksdbForwardIndex.cpp
    invlists/EncodedDocument.cpp
    invlists/IndexWriter.cpp
    invlists/RocksdbInvertedList.cpp
    invlists/ForwardIndexIterator.cpp
    invlists/IndexWriter.cpp
    invlists/InvertedIterator.cpp
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
    query/decode.cpp)

set(LINT_DB_HEADERS
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
    invlists/IndexWriter.h
    invlists/InvertedList.h
    invlists/Iterator.h
    invlists/EncodedDocument.h
    invlists/RocksdbForwardIndex.h
    invlists/InvertedIterator.h
    quantizers/PQDistanceTables.h
    quantizers/impl/product_quantizer.h
    quantizers/CoarseQuantizer.h
    quantizers/impl/kmeans.h
    quantizers/IdentityQuantizer.h
    invlists/RocksdbInvertedList.h
    invlists/ForwardIndexIterator.h
    invlists/IndexWriter.h
    invlists/PostingData.h
    invlists/ContextIterator.h
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
    version.h)

set(LINT_DB_HEADERS
    ${LINT_DB_HEADERS}
    PARENT_SCOPE)

add_library(lintdb_lib ${LINT_DB_SRC})

string(FIND "${CMAKE_CXX_FLAGS}" "FINTEGER" finteger_idx)
if(${finteger_idx} EQUAL -1)
  target_compile_definitions(lintdb_lib PRIVATE FINTEGER=int)
endif()

if(MSVC OR LINUX)
  set(OpenMP_CXX_FLAGS "-fopenmp=libiomp5")
  set(OpenMP_CXX_LIB_NAMES "libiomp5")
  # the below is necessary in github actions when calling pip install .
  cmake_path(GET CMAKE_CURRENT_BINARY_DIR PARENT_PATH BUILD_PARENT_DIR)
  set(OpenMP_libiomp5_LIBRARY "${BUILD_PARENT_DIR}/vcpkg_installed/x64-linux/lib/intel64/libiomp5.so")
  message(STATUS "OpenMP_libiomp5_LIBRARY: ${OpenMP_libiomp5_LIBRARY}")
  if(EXISTS ${OpenMP_libiomp5_LIBRARY})
    message(STATUS "OpenMP_libiomp5_LIBRARY exists")
  else()
    message(STATUS "OpenMP_libiomp5_LIBRARY does not exist")
  endif()
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  target_link_libraries(lintdb_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

message(STATUS "using MKL root: ${MKLROOT}")
find_package(MKL)
if(MKL_FOUND)
  message(STATUS "libraries: ${MKL_LIBRARIES}")
  target_link_libraries(lintdb_lib PRIVATE ${MKL_LIBRARIES})
else()
  find_package(OpenBLAS CONFIG REQUIRED)
  target_link_libraries(lintdb_lib PRIVATE OpenBLAS::OpenBLAS)


#  find_package(LAPACK CONFIG REQUIRED)
#  target_link_libraries(lintdb_lib PRIVATE LAPACK::LAPACK)
endif()

# this is heuristically generated, and may not be correct
find_package(faiss CONFIG REQUIRED)
target_link_libraries(lintdb_lib PRIVATE faiss)

# this is heuristically generated, and may not be correct
find_package(RocksDB CONFIG REQUIRED)
target_link_libraries(lintdb_lib PRIVATE RocksDB::rocksdb)

find_package(flatbuffers CONFIG REQUIRED)
target_link_libraries(lintdb_lib PRIVATE flatbuffers::flatbuffers)

find_package(glog CONFIG REQUIRED)
target_link_libraries(lintdb_lib PRIVATE glog::glog)

# this is heuristically generated, and may not be correct
find_package(Microsoft.GSL CONFIG REQUIRED)
target_link_libraries(lintdb_lib PRIVATE Microsoft.GSL::GSL)

find_package(jsoncpp CONFIG REQUIRED)
target_link_libraries(lintdb_lib PUBLIC JsonCpp::JsonCpp)

find_package(Bitsery CONFIG REQUIRED)
target_link_libraries(lintdb_lib PUBLIC Bitsery::bitsery)

if(MSVC)
  target_compile_options(lintdb_lib PRIVATE /W4 /WX)
else()
  target_compile_options(lintdb_lib PRIVATE -Wall -Wpedantic)
endif()

target_include_directories(lintdb_lib
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(lintdb_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
                                            WINDOWS_EXPORT_ALL_SYMBOLS ON)

foreach(header ${LINT_DB_HEADERS})
  get_filename_component(dir ${header} DIRECTORY)
  install(FILES ${header} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lintdb/${dir})
endforeach()

install(
  TARGETS lintdb_lib
  EXPORT lintdb-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/lintdb-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_file(${PROJECT_SOURCE_DIR}/cmake/lintdb-config.cmake.in
               ${PROJECT_BINARY_DIR}/cmake/lintdb-config.cmake COPYONLY)
install(FILES ${PROJECT_BINARY_DIR}/cmake/lintdb-config.cmake
              ${PROJECT_BINARY_DIR}/cmake/lintdb-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/lintdb)

install(EXPORT lintdb-targets DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/lintdb)
