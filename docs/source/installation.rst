Installation
=============

Conda (Preferred)
-----------------

To use LintDB, we recommend using conda to install:

.. code-block:: console

   $ conda install lintdb


Building from source
--------------------

LintDB is built in C++ using CMake. 

LintDB depends on a few libraries:
- OpenBLAS  
- OpenMP  

We prefer building through conda, which tracks dependencies and build scripts.

.. code-block:: console

   conda install conda-build
   cd conda && conda build lintdb
   conda install --use-local lintdb


Below are methods to build outside of conda on linux.

Linux
-------
Install dependencies
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   sudo apt-get install xorg-dev gfortran libopenblas-dev liblapacke-dev libfftw3-dev cmake unzip libomp swig


Install Clang
^^^^^^^^^^^^^^

On Linux, you can go to the official LLVM website and follow their automatic installation script:
https://apt.llvm.org/

Build LintDB
^^^^^^^^^^^^

The below code will build all the targets including the Python bindings.

.. code-block:: console

   git clone https:://github.com/deployql/lintdb lintdb
   cd lintdb
   mkdir build
   CC=clang CXX=clang++ cmake -S . -B build -DCMAKE_MAKE_PROGRAM=make -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j12  # replace 12 with the number of cores you want to use for building

4. Build Python Bindings

.. code-block:: console

   cmake --build build --target pylintdb -j12  # replace 12 with the number of cores you want to use for building
   cd build/lintdb/python && python setup.py build

