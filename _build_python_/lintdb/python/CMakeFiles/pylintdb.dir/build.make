# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/matt/deployql/LintDB

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/matt/deployql/LintDB/_build_python_

# Include any dependencies generated for this target.
include lintdb/python/CMakeFiles/pylintdb.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lintdb/python/CMakeFiles/pylintdb.dir/compiler_depend.make

# Include the progress variables for this target.
include lintdb/python/CMakeFiles/pylintdb.dir/progress.make

# Include the compile flags for this target's objects.
include lintdb/python/CMakeFiles/pylintdb.dir/flags.make

lintdb/python/CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o: lintdb/python/CMakeFiles/pylintdb.dir/flags.make
lintdb/python/CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o: lintdb/python/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx
lintdb/python/CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o: lintdb/python/CMakeFiles/pylintdb.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/matt/deployql/LintDB/_build_python_/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lintdb/python/CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o"
	cd /home/matt/deployql/LintDB/_build_python_/lintdb/python && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lintdb/python/CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o -MF CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o.d -o CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o -c /home/matt/deployql/LintDB/_build_python_/lintdb/python/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx

lintdb/python/CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.i"
	cd /home/matt/deployql/LintDB/_build_python_/lintdb/python && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matt/deployql/LintDB/_build_python_/lintdb/python/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx > CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.i

lintdb/python/CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.s"
	cd /home/matt/deployql/LintDB/_build_python_/lintdb/python && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matt/deployql/LintDB/_build_python_/lintdb/python/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx -o CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.s

# Object files for target pylintdb
pylintdb_OBJECTS = \
"CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o"

# External object files for target pylintdb
pylintdb_EXTERNAL_OBJECTS =

lintdb/python/_pylintdb.so: lintdb/python/CMakeFiles/pylintdb.dir/CMakeFiles/pylintdb.dir/lintdbPYTHON_wrap.cxx.o
lintdb/python/_pylintdb.so: lintdb/python/CMakeFiles/pylintdb.dir/build.make
lintdb/python/_pylintdb.so: lintdb/liblintdb.so
lintdb/python/_pylintdb.so: /usr/lib/llvm-18/lib/libomp.so
lintdb/python/_pylintdb.so: /lib/x86_64-linux-gnu/libpthread.a
lintdb/python/_pylintdb.so: lintdb/python/CMakeFiles/pylintdb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/matt/deployql/LintDB/_build_python_/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library _pylintdb.so"
	cd /home/matt/deployql/LintDB/_build_python_/lintdb/python && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pylintdb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lintdb/python/CMakeFiles/pylintdb.dir/build: lintdb/python/_pylintdb.so
.PHONY : lintdb/python/CMakeFiles/pylintdb.dir/build

lintdb/python/CMakeFiles/pylintdb.dir/clean:
	cd /home/matt/deployql/LintDB/_build_python_/lintdb/python && $(CMAKE_COMMAND) -P CMakeFiles/pylintdb.dir/cmake_clean.cmake
.PHONY : lintdb/python/CMakeFiles/pylintdb.dir/clean

lintdb/python/CMakeFiles/pylintdb.dir/depend:
	cd /home/matt/deployql/LintDB/_build_python_ && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matt/deployql/LintDB /home/matt/deployql/LintDB/lintdb/python /home/matt/deployql/LintDB/_build_python_ /home/matt/deployql/LintDB/_build_python_/lintdb/python /home/matt/deployql/LintDB/_build_python_/lintdb/python/CMakeFiles/pylintdb.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lintdb/python/CMakeFiles/pylintdb.dir/depend

