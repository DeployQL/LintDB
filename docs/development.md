# Development

## LintDB C++ Libraries
To develop on LintDB, there are a few dependencies that you need to install. The below instructions are for Ubuntu.

### [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/overview)
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh
```

### [clang](https://apt.llvm.org/)
We expect clang as the compiler. This helps align with our expectations of MKL libraries detailed below.
```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh all
```

### [miniforge](https://github.com/conda-forge/miniforge) (recommended)
Miniforge is a minimal installer for conda that automatically installs conda-forge packages.

We can create an isolated environment for lintdb development.
```bash
conda create -n lintdb python=3.10
conda activate lintdb
```

### Recommended Python Libraries
There are a few helpful python libraries that are used in profiling and testing LintDB.
```bash
pip install graph2dot
```
---
# Python LintDB

In addition to the above, developing with the Python LintDB library requires a few more dependencies.

LintDB uses nanobind to create Python bindings. It also comes with a helpful CLI tool to create stubs for Python.

```bash
pip install nanobind
```

### creating python stubs
```bash
python -m nanobind.stubgen -m lintdb.core -M py.typed -o core.pyi 
```

---

# Makefile commands

The Makefile at the root of the repository has a few commands that can help you get started.
CMakePresets.json is used to configure the build system.

```bash
# build a debug target with tests.
make build-debug

# build a release target
make build-release

# run tests
make tests

# run benchmarks
make benchmarks

# profile LintDB (note some variables need to change in the Makefile)
make callgrind
```

---

You'll notice that each target is statically linked. However, we dynamically depend on finding either MKL or OpenBLAS at runtime.

## MKL vs OpenBLAS

LintDB currently uses either MKL or OpenBLAS for linear algebra operations. By default, we use MKL on Windows and Ubuntu. On MacOS, we use OpenBLAS. 

It should be noted that MKL doesn't always play well with OpenMP. We specify linking against intel's version of OpenMP, but
at runtime, it's possible we find a different version. This can lead to performance issues.

It can be helpful to refer to [Intel's threading layer documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/dynamic-select-the-interface-and-threading-layer.html) and
try `INTEL` or `GNU`. Running `ldd path/to/liblintdb_lib.so` will output what libraries are being linked at runtime to verify if there
are issues.