set(LINTDB_SERVER_SRCS
        main.cpp
        controllers/v1/query_node_translator.h
        controllers/v1/result_translator.h
)
add_executable(lintdb-server ${LINTDB_SERVER_SRCS})

find_package(Drogon CONFIG REQUIRED)
target_link_libraries(lintdb-server PRIVATE Drogon::Drogon)

find_package(args CONFIG REQUIRED)
target_link_libraries(lintdb-server PRIVATE taywee::args)

target_link_libraries(lintdb-server PRIVATE lintdb_lib)


install(
        TARGETS lintdb-server
        EXPORT lintdb-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})