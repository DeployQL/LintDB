# FindMLAS.cmake

# Define the module interface variables
set(MLAS_INCLUDE_DIRS "")
set(MLAS_LIBRARIES "")

# FetchContent mechanism
include(FetchContent)

# Fetch onnxruntime if not already fetched
if (NOT onnxruntime_POPULATED)
    FetchContent_Declare(
            onnxruntime
            GIT_REPOSITORY https://github.com/microsoft/onnxruntime.git
            GIT_TAG v1.16.0 # Replace with the specific version you need
    )
    FetchContent_MakeAvailable(onnxruntime)
endif()

# Define paths based on fetched content
set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})

# Find the MLAS include directory
find_path(MLAS_INCLUDE_DIR
        NAMES mlas.h
        PATHS ${ONNXRUNTIME_ROOT}/onnxruntime/core/mlas/inc
        NO_DEFAULT_PATH
)

# Find the MLAS library
find_library(MLAS_LIBRARY
        NAMES mlas
        PATHS ${ONNXRUNTIME_ROOT}/onnxruntime/core/mlas/lib
        NO_DEFAULT_PATH
)

# Set the found variables
if (MLAS_INCLUDE_DIR AND MLAS_LIBRARY)
    set(MLAS_INCLUDE_DIRS ${MLAS_INCLUDE_DIR})
    set(MLAS_LIBRARIES ${MLAS_LIBRARY})
else()
    message(FATAL_ERROR "MLAS library not found.")
endif()

# Provide usage information
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLAS DEFAULT_MSG MLAS_INCLUDE_DIR MLAS_LIBRARY)

# Define an IMPORTED target for modern CMake usage
if (NOT TARGET MLAS::MLAS)
    add_library(MLAS::MLAS INTERFACE IMPORTED)
    set_target_properties(MLAS::MLAS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MLAS_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${MLAS_LIBRARIES}"
    )
endif()
