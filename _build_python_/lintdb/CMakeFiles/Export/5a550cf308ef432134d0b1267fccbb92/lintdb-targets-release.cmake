#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lintdb" for configuration "Release"
set_property(TARGET lintdb APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(lintdb PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liblintdb.so"
  IMPORTED_SONAME_RELEASE "liblintdb.so"
  )

list(APPEND _cmake_import_check_targets lintdb )
list(APPEND _cmake_import_check_files_for_lintdb "${_IMPORT_PREFIX}/lib/liblintdb.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
