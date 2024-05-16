Benchmarks
=============

Mac OS
-------

Benchmark: LoTTE Lifestyle 40k
-------------------------------
Details:
- 40k documents
- Dev split
- Search data

Hardware:
- M1 Macbook Air


colbert:

.. code-block:: console

    Average search latency: 0.03s
    Median search latency: 0.03s
    95th percentile search latency: 0.03s
    99th percentile search latency: 0.06s

lintdb:

.. code-block:: console

    Average search latency: 0.02s
    Median search latency: 0.02s
    95th percentile search latency: 0.02s
    99th percentile search latency: 0.02s



Linux
------

Benchmark: LoTTE Lifestyle 40k
-------------------------------
Details:
- 40k documents
- Dev split
- Search data

Hardware:
Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz

.. code-block:: console

    2024-04-29T03:30:14+00:00
    Running ./build_benchmarks/benchmarks/bench_lintdb
    Run on (48 X 3500 MHz CPU s)
    CPU Caches:
    L1 Data 32 KiB (x24)
    L1 Instruction 32 KiB (x24)
    L2 Unified 256 KiB (x24)
    L3 Unified 30720 KiB (x2)
    Load Average: 0.10, 2.13, 4.10
    ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I20240429 03:30:14.280722 266773 index.cpp:40] loading LintDB from path: experiments/py_index_bench_colbert-lifestyle-2024-04-03
    I20240429 03:30:14.675765 266773 index.cpp:40] loading LintDB from path: experiments/py_index_bench_colbert-lifestyle-2024-04-03
    I20240429 03:30:15.496747 266773 index.cpp:40] loading LintDB from path: experiments/py_index_bench_colbert-lifestyle-2024-04-03
    ------------------------------------------------------------------
    Benchmark                        Time             CPU   Iterations
    ------------------------------------------------------------------
    BM_lintdb_search_mean         40.2 ms         40.2 ms          212
    BM_lintdb_search_median       38.9 ms         38.9 ms          212
    BM_lintdb_search_stddev       4.51 ms         4.51 ms          212
    BM_lintdb_search_cv          11.21 %         11.21 %           212