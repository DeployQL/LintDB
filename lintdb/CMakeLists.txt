
set(
    LINT_DB_SRC
    index.cpp
    quantizers/Binarizer.cpp
    Encoder.cpp
    quantizers/ProductEncoder.cpp
    quantizers/io.cpp
    util.cpp
    invlists/RocksdbList.cpp
    invlists/util.cpp
    invlists/EncodedDocument.cpp
    retriever/PlaidRetriever.cpp
    retriever/plaid.cpp
    retriever/EMVBRetriever.cpp
    schema/util.cpp
)

set(
    LINT_DB_HEADERS
    index.h
    api.h
    assert.h
    exception.h
    constants.h
    cf.h
    RawPassage.h
    EmbeddingBlock.h
    Encoder.h
    quantizers/Binarizer.h
    quantizers/Quantizer.h
    quantizers/ProductEncoder.h
    quantizers/io.h
    SearchOptions.h
    SearchResult.h
    util.h
    invlists/InvertedList.h
    invlists/Iterator.h
    invlists/EncodedDocument.h
    invlists/RocksdbList.h
    invlists/util.h
    retriever/Retriever.h
    retriever/PlaidRetriever.h
    retriever/plaid.h
    retriever/EMVBRetriever.h
    retriever/emvb_util.h
    retriever/emvb.h
    retriever/emvb_generic.h
    retriever/emvb_avx.h
    schema/forward_index_generated.h
    schema/inverted_index_generated.h
    schema/mapping_generated.h
    schema/util.h
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

message(STATUS "using MKL root: ${MKLROOT}")
find_package(MKL)
if (MKL_FOUND)
  # target_link_libraries(faiss PRIVATE "-Wl,--start-group" ${MKL_LIBRARIES} "-Wl,--end-group")
  # target_link_libraries(lintdb PRIVATE MKL::MKL)
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

install(TARGETS lintdb
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
