from lintdb.core import (
    Schema,
    ColbertField,
    StoredField,
    IndexedField,
    ContextField,
    QuantizerType,
    Binarizer,
    Configuration,
    CoarseQuantizer,
    SearchOptions,
    DataType,
    FieldValue,
    TensorFieldValue,
    QuantizedTensorFieldValue,
    IntFieldValue,
    FloatFieldValue,
    TextFieldValue,
    DateFieldValue,
    Document,
    IndexIVF,
    Version,
    TermQueryNode,
    VectorQueryNode,
    AndQueryNode,
    Query
)
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
def single_search(dataset:str='lifestyle', split:str='dev',profile=False, checkpoint:str='colbert-ir/colbertv2.0', index_path:str='experiments/py_index_bench_test-collection-xtr'):
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
        opts = ldb.SearchOptions()
        results = index.search(
            0,
            converted, 
            32, # nprobe
            100, # k to return
            opts
        )
        latencies.append((time.perf_counter() - start)*1000)
        if profile:
            callgrind_stop_instrumentation()
            callgrind_dump_stats("callgrind.out.single_search")
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
def index(dataset:str='lifestyle', split:str='dev',profile=False, checkpoint:str='colbert-ir/colbertv2.0', index_path:str='experiments/py_index_bench_colbert-lifestyle-2024-04-03'):
    latencies = []
    num_docs=1000

    with tempfile.TemporaryDirectory(prefix="lintdb_bench_collection") as dir_one:
        schema = Schema(
            [
                ColbertField('colbert', DataType.TENSOR, {
                    'dimensions': 128,
                    'quantization': QuantizerType.BINARIZER,
                    "num_centroids": 2,
                    "num_iterations": 12,
                })
            ]
        )
        config = Configuration()
        index = IndexIVF(dir_one, schema, config)

        # create random embeddings
        embeddings = np.ones((120 * num_docs, 128)).astype('float32')

        training_docs = []
        for i in range(num_docs):
            embeds = embeddings[i*120:(i+1)*120]
            doc = Document(i, [TensorFieldValue('colbert', embeds)])
            training_docs.append(doc)

        index.train(training_docs)

        count=0
        for id in range(1000):
            
            start = time.perf_counter()

            index.add(0, [training_docs[id]])

            latencies.append((time.perf_counter() - start)*1000)

            if count == 212:
                break

    
    print(f"Average add latency: {np.mean(latencies):.2f}ms")
    print(f"Median add latency: {np.median(latencies):.2f}ms")
    print(f"95th percentile add latency: {np.percentile(latencies, 95):.2f}ms")
    print(f"99th percentile add latency: {np.percentile(latencies, 99):.2f}ms")

    
if __name__ == "__main__":
    app()