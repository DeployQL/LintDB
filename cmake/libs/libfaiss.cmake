#file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/third_party/faiss/faiss/**/*.cpp")

file(
        GLOB FAISS_SOURCES ${PROJECT_SOURCE_DIR}/third_party/faiss/faiss/*.cpp
        ${PROJECT_SOURCE_DIR}/third_party/faiss/faiss/impl/*.cpp
        ${PROJECT_SOURCE_DIR}/third_party/faiss/faiss/invlists/*.cpp
        ${PROJECT_SOURCE_DIR}/third_party/faiss/faiss/utils/*.cpp
        ${PROJECT_SOURCE_DIR}/third_party/faiss/faiss/utils/distances_fused/*.cpp
)

file(GLOB AVX512_SOURCES third_party/faiss/faiss/*avx512*.cpp)
list(REMOVE_ITEM FAISS_SOURCES ${AVX512_SOURCES})


# remove RHNSW as per milvus
file(GLOB RHNSW_SOURCES third_party/faiss/faiss/impl/RHNSW.cpp)
list(REMOVE_ITEM FAISS_SOURCES ${RHNSW_SOURCES})

find_package(BLAS REQUIRED)


#if(LINUX)

#    list(REMOVE_ITEM SOURCES ${AVX2_SOURCES})

    file(GLOB NEON_SOURCES third_party/faiss/faiss/impl/*neon*.cpp)
    list(REMOVE_ITEM FAISS_SOURCES ${NEON_SOURCES})

    add_library(faiss STATIC ${FAISS_SOURCES})

    target_compile_options(
            faiss
            PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -mfma
            -mf16c
            -mavx2
            -msse4.2
            -mpopcnt
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)

#find_package(OpenMP REQUIRED)


message(STATUS "faiss using BLAS: ${BLAS_LIBRARIES}")
target_link_libraries(
        faiss PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
)

    target_compile_definitions(faiss PRIVATE FINTEGER=int)
#endif()