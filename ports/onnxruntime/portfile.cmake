vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

set(VERSION 1.17.3)
set(ONNX_FILENAME onnxruntime-win-x64-gpu-${VERSION})
if (MSVC)
vcpkg_download_distfile(ARCHIVE
    URLS "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-win-x64-${VERSION}.zip"
    FILENAME "onnxruntime-win-x64-gpu-${VERSION}.zip"
    SHA512 d9f7c21b0e4ee64e84923904e05d04686231ab9240724dad7e7efc05d890d73404d92e9d07f14d14507d897da846a754b474b7b036e8416a06daaf200e1ec488
)
elseif(UNIX AND NOT APPLE)
vcpkg_download_distfile(ARCHIVE
    URLS "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-x64-${VERSION}.tgz"
    FILENAME "onnxruntime-linux-x64-${VERSION}.tgz"
    SHA512 c13273acb7730f0f5eed569cff479d34c9674f5f39d2a76a2c960835560e9706fd92e07071dd66fe242738c31f0df19d830b7e5083378c9e0657685727725ca0
)
set(ONNX_FILENAME onnxruntime-linux-x64-${VERSION})
elseif(APPLE AND VCPKG_TARGET_ARCHITECTURE MATCHES "x64")
vcpkg_download_distfile(ARCHIVE
    URLS "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-osx-x86_64-${VERSION}.tgz"
    FILENAME "onnxruntime-osx-x86_64-${VERSION}.tgz"
    SHA512 175712dccb8d57cf4f0e7668f3e7ed42329ace19c54f3a5670e8cf13a335faf90889b6c248e855ab3d8ebb1254c6484fe91f7bb732f959816c02463c6b9a9626
)
set(ONNX_FILENAME onnxruntime-osx-x86_64-${VERSION})
elseif(APPLE AND VCPKG_TARGET_ARCHITECTURE MATCHES "arm64")
vcpkg_download_distfile(ARCHIVE
    URLS "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-osx-arm64-${VERSION}.tgz"
    FILENAME "onnxruntime-osx-arm64-${VERSION}.tgz"
    SHA512 1e002f8d2d89cb99d2bd9c2c61ef7cfe4e72724f21a6a3d5df6524f92cc9dd5096754e871b2ee7e5588d5f09a320f5eb0f484a95ff70d4b05990dfa388c344bf
)
set(ONNX_FILENAME onnxruntime-osx-arm64-${VERSION})
endif()

vcpkg_extract_source_archive(
    SOURCE_PATH
    ARCHIVE "${ARCHIVE}"
    NO_REMOVE_ONE_LEVEL
)

file(MAKE_DIRECTORY
        ${CURRENT_PACKAGES_DIR}/include
        ${CURRENT_PACKAGES_DIR}/lib
        ${CURRENT_PACKAGES_DIR}/bin
        ${CURRENT_PACKAGES_DIR}/debug/lib
        ${CURRENT_PACKAGES_DIR}/debug/bin
    )

# copy the include dir to our package
file(COPY
        ${SOURCE_PATH}/${ONNX_FILENAME}/include
        DESTINATION ${CURRENT_PACKAGES_DIR}
    )

# now copy the lib files depending on the platform
if (MSVC)
    file(GLOB_RECURSE ONNX_LIBS ${SOURCE_PATH}/${ONNX_FILENAME}/lib/*.lib)
    file(GLOB_RECURSE ONNX_PDB ${SOURCE_PATH}/${ONNX_FILENAME}/lib/*.pdb)
    file(GLOB_RECURSE ONNX_DLLS ${SOURCE_PATH}/${ONNX_FILENAME}/lib/*.dll)

    file(COPY
            ${SOURCE_PATH}/${ONNX_FILENAME}/include
            DESTINATION ${CURRENT_PACKAGES_DIR}
        )

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime.lib
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime.lib
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime.pdb
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime.pdb
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_cuda.lib
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_cuda.lib
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_cuda.pdb
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_cuda.pdb
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_shared.pdb
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_shared.pdb
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_tensorrt.pdb
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_tensorrt.pdb
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_shared.lib
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_shared.lib
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_tensorrt.lib
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_tensorrt.lib
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_shared.dll
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_shared.dll
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime.dll
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime.dll
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_tensorrt.dll
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_tensorrt.dll
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_cuda.dll
        DESTINATION ${CURRENT_PACKAGES_DIR}/bin)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/onnxruntime_providers_cuda.dll
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/bin)
elseif(UNIX AND NOT APPLE)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.so
    DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.so
    DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.so.${VERSION}
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.so.${VERSION}
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)
elseif(APPLE AND VCPKG_TARGET_ARCHITECTURE MATCHES "x64")
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.dylib
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.dylib
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.${VERSION}.dylib
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.${VERSION}.dylib
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.${VERSION}.dylib.dSYM
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.${VERSION}.dylib.dSYM
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)
elseif(APPLE AND VCPKG_TARGET_ARCHITECTURE MATCHES "arm64")
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.dylib
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.dylib
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.${VERSION}.dylib
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.${VERSION}.dylib
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)

    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.${VERSION}.dylib.dSYM
        DESTINATION ${CURRENT_PACKAGES_DIR}/lib)
    file(COPY ${SOURCE_PATH}/${ONNX_FILENAME}/lib/libonnxruntime.${VERSION}.dylib.dSYM
        DESTINATION ${CURRENT_PACKAGES_DIR}/debug/lib)
endif()
# # Handle copyright
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/${ONNX_FILENAME}/LICENSE")
