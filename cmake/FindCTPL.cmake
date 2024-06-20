# FindCTPL.cmake

# Define the module interface variables
set(CTPL_INCLUDE_DIRS "")

# FetchContent mechanism
include(FetchContent)

# Fetch ctpl if not already fetched
if (NOT ctpl_POPULATED)
    FetchContent_Declare(
            ctpl
            GIT_REPOSITORY https://github.com/vit-vit/CTPL.git
            GIT_TAG main # Replace with the specific tag/branch you need
    )
    FetchContent_MakeAvailable(ctpl)
endif()

# Define paths based on fetched content
set(CTPL_ROOT ${ctpl_SOURCE_DIR})

# Find the CTPL include directory
find_path(CTPL_INCLUDE_DIR
        NAMES ctpl.h ctpl_stl.h
        PATHS ${CTPL_ROOT}
        NO_DEFAULT_PATH
)

# Set the found variables
if (CTPL_INCLUDE_DIR)
    set(CTPL_INCLUDE_DIRS ${CTPL_INCLUDE_DIR})
else()
    message(FATAL_ERROR "CTPL library not found.")
endif()

# Provide usage information
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CTPL DEFAULT_MSG CTPL_INCLUDE_DIR)

# Define an IMPORTED target for modern CMake usage
if (NOT TARGET CTPL::CTPL)
    add_library(CTPL::CTPL INTERFACE IMPORTED)
    set_target_properties(CTPL::CTPL PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CTPL_INCLUDE_DIRS}"
    )
endif()
