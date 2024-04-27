import lintdb as ldb
from datasets import load_dataset
from collections import namedtuple
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import os
import sys
import jsonlines
from collections import defaultdict
from tqdm import tqdm
import time
import numpy as np
import typer
import random 
from typing import List, Annotated
from common import get_memory_usage
from lotte.common import _evaluate_dataset, load_lotte

try:
    from valgrind import callgrind_start_instrumentation, callgrind_stop_instrumentation, callgrind_dump_stats
except ImportError:
    print("didn't find valgrind")
    def callgrind_stop_instrumentation():
        pass

    def callgrind_start_instrumentation():
        pass

    def callgrind_dump_stats(path:str):
        pass


app = typer.Typer()

@app.command()
def single_search(dataset:str='lifestyle', split:str='dev',profile=False, checkpoint:str='colbert-ir/colbertv2.0', index_path:str='experiments/py_index_bench_colbert-lifestyle-2024-04-03'):
    latencies = []
    memory = []

    index = ldb.IndexIVF(index_path)
    rankings = {}

    count=0
    for id in range(1000):
        embeddings = np.ones((32, 128)).astype('float32')
        converted = embeddings

        start = time.perf_counter()
        if profile:
            callgrind_start_instrumentation()
        results = index.search(
            0,
            converted, 
            32, # nprobe
            100, # k to return
        )
        if profile:
            callgrind_stop_instrumentation()
            callgrind_dump_stats("callgrind.out.single_search")
        latencies.append((time.perf_counter() - start)*1000)
        memory.append(get_memory_usage())
        rankings[id] = [x.id for x in results]
        count+=1
        # if count == 2:
        #     break

        # Stats(pr).strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
    # _evaluate_dataset(rankings, dataset, 'search', k=5)

    
    print(f"Average search latency: {np.mean(latencies):.2f}ms")
    print(f"Median search latency: {np.median(latencies):.2f}ms")
    print(f"95th percentile search latency: {np.percentile(latencies, 95):.2f}ms")
    print(f"99th percentile search latency: {np.percentile(latencies, 99):.2f}ms")

    print(f"Average memory usage: {np.mean(memory):.2f}MB")
    print(f"Median memory usage: {np.median(memory):.2f}MB")
    print(f"95th percentile memory usage: {np.percentile(memory, 95):.2f}MB")
    print(f"99th percentile memory usage: {np.percentile(memory, 99):.2f}MB")

    
if __name__ == "__main__":
    app()