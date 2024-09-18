

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

find_package(Python
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule
)

message("=== Selected Python Variables ===")
message(STATUS "Python3_STDLIB: " ${Python_STDLIB})
message(STATUS "Python3_SITELIB: " ${Python_SITELIB})
message(STATUS "Python3_VERSION: " ${Python_VERSION})
message(STATUS "Python3_EXECUTABLE: " ${Python_EXECUTABLE})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/nanobind nanobind EXCLUDE_FROM_ALL)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
        core
        STABLE_ABI
        pylintdb.cpp
)

target_link_libraries(core PRIVATE
        lintdb
        faiss
)
target_include_directories(core PRIVATE nanobind)


install(TARGETS core LIBRARY DESTINATION lintdb)

# this doesn't work because python looks for typing_extensions and doesn't find it on py3.10
#nanobind_add_stub(
#        lintdb_stub
#        INSTALL_TIME
#        MODULE core
#        OUTPUT core.pyi
#        PYTHON_PATH $<TARGET_FILE_DIR:core>
#        DEPENDS core
#        VERBOSE
#)
#
#install(FILES "core.pyi" DESTINATION lintdb)
#install(FILES "py.typed" DESTINATION lintdb)