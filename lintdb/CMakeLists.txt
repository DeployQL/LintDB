
set(
    LINT_DB_SRC
    index.cpp
    plaid.cpp
    invlists/RocksdbList.cpp
    invlists/util.cpp
    invlists/EncodedDocument.cpp
    schema/util.cpp
)
# file(GLOB LINT_DB_SRC "*.cpp" "invlist/*.cpp" "schema/*.cpp")
# file(GLOB LINT_DB_HEADERS "*.h" "invlist/*.h" "schema/*.h")

set(
    LINT_DB_HEADERS
    index.h
    api.h
    assert.h
    exception.h
    plaid.h
    constants.h
    cf.h
    EmbeddingBlock.h
    invlists/InvertedList.h
    invlists/Iterator.h
    invlists/EncodedDocument.h
    invlists/RocksdbList.h
    invlists/util.h
    schema/forward_index_generated.h
    schema/inverted_index_generated.h
    schema/util.h
)

set(LINT_DB_HEADERS ${LINT_DB_HEADERS} PARENT_SCOPE)

add_library(lintdb ${LINT_DB_SRC})

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

