
set(
    LINT_DB_SRC
    index.cpp
    plaid.cpp
    Binarizer.cpp
    Encoder.cpp
    util.cpp
    invlists/RocksdbList.cpp
    invlists/util.cpp
    invlists/EncodedDocument.cpp
    schema/util.cpp
)

set(
    LINT_DB_HEADERS
    index.h
    api.h
    assert.h
    exception.h
    plaid.h
    constants.h
    cf.h
    RawPassage.h
    EmbeddingBlock.h
    Binarizer.h
    Encoder.h
    util.h
    invlists/InvertedList.h
    invlists/Iterator.h
    invlists/EncodedDocument.h
    invlists/RocksdbList.h
    invlists/util.h
    schema/forward_index_generated.h
    schema/inverted_index_generated.h
    schema/mapping_generated.h
    schema/util.h
)

set(LINT_DB_HEADERS ${LINT_DB_HEADERS} PARENT_SCOPE)

add_library(lintdb ${LINT_DB_SRC})

if(NOT WIN32)
  target_compile_options(lintdb PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
else()
  # MSVC enables FMA with /arch:AVX2; no separate flags for F16C, POPCNT
  # Ref. FMA (under /arch:AVX2): https://docs.microsoft.com/en-us/cpp/build/reference/arch-x64
  # Ref. F16C (2nd paragraph): https://walbourn.github.io/directxmath-avx2/
  # Ref. POPCNT: https://docs.microsoft.com/en-us/cpp/intrinsics/popcnt16-popcnt-popcnt64
  target_compile_options(lintdb PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  # we need bigobj for the swig wrapper
  add_compile_options(/bigobj)
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

add_subdirectory(python)

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

add_executable(
  lintdb-bin
  ${PROJECT_SOURCE_DIR}/tests/search_bin.cpp
)

target_link_libraries(lintdb-bin PRIVATE lintdb)
find_package(faiss CONFIG REQUIRED)
target_link_libraries(lintdb-bin PRIVATE faiss)
# target_link_libraries(lintdb-bin PRIVATE OpenBLAS::OpenBLAS)
# target_link_libraries(lintdb-bin PRIVATE ${LAPACK_LIBRARIES})
# target_link_libraries(lintdb-bin PRIVATE Microsoft.GSL::GSL)