import lintdb as ldb

import time
import numpy as np
import typer
import random 
import tempfile
from typing import List, Annotated
from common import get_memory_usage

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
        # if profile:
        #     callgrind_start_instrumentation()
        opts = ldb.SearchOptions()
        results = index.search(
            0,
            converted, 
            32, # nprobe
            100, # k to return
            opts
        )
        latencies.append((time.perf_counter() - start)*1000)
        # if profile:
        #     callgrind_stop_instrumentation()
        #     callgrind_dump_stats("callgrind.out.single_search")
        memory.append(get_memory_usage())
        rankings[id] = [x.id for x in results]
        count+=1
        if count == 212:
            break

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


@app.command()
def collection(dataset:str='lifestyle', split:str='dev',profile=False, checkpoint:str='colbert-ir/colbertv2.0', index_path:str='experiments/py_index_bench_colbert-lifestyle-2024-04-03'):
    latencies = []
    memory = []

    with tempfile.TemporaryDirectory(prefix="lintdb_bench_collection") as dir_one:
        index = ldb.IndexIVF(index_path, 32768, 128, 2, 6, 16, ldb.IndexEncoding_BINARIZER)
        opts = ldb.CollectionOptions()
        opts.model_file = 'assets/model.onnx'
        opts.tokenizer_file = 'assets/colbert_tokenizer.json'
        collection = ldb.Collection(index, opts)
        collection.train(['hello world!'] * 35000)
        rankings = {}

        count=0
        for id in range(1000):
            
            start = time.perf_counter()
            collection.add(0, id, "hello world!", {"title": "metadata"})
            latencies.append((time.perf_counter() - start)*1000)

            memory.append(get_memory_usage())

            if count == 212:
                break

    
    print(f"Average add latency: {np.mean(latencies):.2f}ms")
    print(f"Median add latency: {np.median(latencies):.2f}ms")
    print(f"95th percentile add latency: {np.percentile(latencies, 95):.2f}ms")
    print(f"99th percentile add latency: {np.percentile(latencies, 99):.2f}ms")

    
if __name__ == "__main__":
    app()