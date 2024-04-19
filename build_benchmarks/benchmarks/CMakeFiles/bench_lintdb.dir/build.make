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
CMAKE_BINARY_DIR = /home/matt/deployql/LintDB/build_benchmarks

# Include any dependencies generated for this target.
include benchmarks/CMakeFiles/bench_lintdb.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include benchmarks/CMakeFiles/bench_lintdb.dir/compiler_depend.make

# Include the progress variables for this target.
include benchmarks/CMakeFiles/bench_lintdb.dir/progress.make

# Include the compile flags for this target's objects.
include benchmarks/CMakeFiles/bench_lintdb.dir/flags.make

benchmarks/CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o: benchmarks/CMakeFiles/bench_lintdb.dir/flags.make
benchmarks/CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o: /home/matt/deployql/LintDB/benchmarks/bench_lintdb.cpp
benchmarks/CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o: benchmarks/CMakeFiles/bench_lintdb.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/matt/deployql/LintDB/build_benchmarks/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object benchmarks/CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o"
	cd /home/matt/deployql/LintDB/build_benchmarks/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT benchmarks/CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o -MF CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o.d -o CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o -c /home/matt/deployql/LintDB/benchmarks/bench_lintdb.cpp

benchmarks/CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.i"
	cd /home/matt/deployql/LintDB/build_benchmarks/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matt/deployql/LintDB/benchmarks/bench_lintdb.cpp > CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.i

benchmarks/CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.s"
	cd /home/matt/deployql/LintDB/build_benchmarks/benchmarks && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matt/deployql/LintDB/benchmarks/bench_lintdb.cpp -o CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.s

# Object files for target bench_lintdb
bench_lintdb_OBJECTS = \
"CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o"

# External object files for target bench_lintdb
bench_lintdb_EXTERNAL_OBJECTS =

benchmarks/bench_lintdb: benchmarks/CMakeFiles/bench_lintdb.dir/bench_lintdb.cpp.o
benchmarks/bench_lintdb: benchmarks/CMakeFiles/bench_lintdb.dir/build.make
benchmarks/bench_lintdb: lintdb/liblintdb.so
benchmarks/bench_lintdb: vcpkg_installed/x64-linux/lib/libbenchmark.a
benchmarks/bench_lintdb: vcpkg_installed/x64-linux/lib/libbenchmark_main.a
benchmarks/bench_lintdb: /usr/lib/llvm-18/lib/libomp.so
benchmarks/bench_lintdb: /lib/x86_64-linux-gnu/libpthread.a
benchmarks/bench_lintdb: vcpkg_installed/x64-linux/lib/libbenchmark.a
benchmarks/bench_lintdb: benchmarks/CMakeFiles/bench_lintdb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/matt/deployql/LintDB/build_benchmarks/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bench_lintdb"
	cd /home/matt/deployql/LintDB/build_benchmarks/benchmarks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_lintdb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmarks/CMakeFiles/bench_lintdb.dir/build: benchmarks/bench_lintdb
.PHONY : benchmarks/CMakeFiles/bench_lintdb.dir/build

benchmarks/CMakeFiles/bench_lintdb.dir/clean:
	cd /home/matt/deployql/LintDB/build_benchmarks/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/bench_lintdb.dir/cmake_clean.cmake
.PHONY : benchmarks/CMakeFiles/bench_lintdb.dir/clean

benchmarks/CMakeFiles/bench_lintdb.dir/depend:
	cd /home/matt/deployql/LintDB/build_benchmarks && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matt/deployql/LintDB /home/matt/deployql/LintDB/benchmarks /home/matt/deployql/LintDB/build_benchmarks /home/matt/deployql/LintDB/build_benchmarks/benchmarks /home/matt/deployql/LintDB/build_benchmarks/benchmarks/CMakeFiles/bench_lintdb.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : benchmarks/CMakeFiles/bench_lintdb.dir/depend

