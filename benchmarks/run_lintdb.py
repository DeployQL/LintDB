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
# import cProfile
# from pstats import SortKey, Stats

app = typer.Typer()

LoTTeDataset = namedtuple('LoTTeDataset', ['collection', 'queries', 'qids', 'dids'])


def load_lotte(dataset, split, filter=False, start=0, stop=500000):
    collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
    collection = [x['text'] for x in collection_dataset[split + '_collection']]
    dids = [x['doc_id'] for x in collection_dataset[split + '_collection']]

    queries_dataset = load_dataset("colbertv2/lotte", dataset)
    queries = [x['query'] for x in queries_dataset['search_' + split]]
    qids = [x['qid'] for x in queries_dataset['search_' + split]]

    f'Loaded {len(queries)} queries and {len(collection):,} passages'

    if not filter:
        return LoTTeDataset(collection, queries, qids, dids)
    else:
        answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + split]]
        filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(start <= x < stop for x in apids)]
        filtered_qids = [i for i,(q, apids) in enumerate(zip(queries, answer_pids)) if any(start <= x < stop for x in apids)]
        filtered_dids = [x for x in dids if start <= x < stop]
        f'Filtered down to {len(filtered_queries)} queries'

        return LoTTeDataset(collection[start:stop], filtered_queries, filtered_qids, filtered_dids)
    

@app.command()
def single_search(dataset:str='lifestyle', split:str='dev', checkpoint:str='colbert-ir/colbertv2.0', index_path:str='experiments/py_index_bench_colbert-lifestyle-2024-03-20'):
    checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
    config = ColBERTConfig.from_existing(checkpoint_config, None)

    from colbert.modeling.checkpoint import Checkpoint
    from colbert import Searcher
    checkpoint = Checkpoint(checkpoint, config)

    d = load_lotte(dataset, split, stop=40000000)
    latencies = []
    memory = []

    index = ldb.IndexIVF(index_path)

    # with cProfile.Profile() as pr:
    for query in d.queries:
        embeddings = checkpoint.queryFromText([query])
        converted = np.squeeze(embeddings.numpy().astype('float32'))

        start = time.perf_counter()
        results = index.search(
            0,
            converted, 
            64, # nprobe
            100, # k to return
        )
        latencies.append(time.perf_counter() - start)
        memory.append(get_memory_usage())
        break

        # Stats(pr).strip_dirs().sort_stats(SortKey.TIME).print_stats(10)

    
    print(f"Average search latency: {np.mean(latencies):.2f}s")
    print(f"Median search latency: {np.median(latencies):.2f}s")
    print(f"95th percentile search latency: {np.percentile(latencies, 95):.2f}s")
    print(f"99th percentile search latency: {np.percentile(latencies, 99):.2f}s")

    print(f"Average memory usage: {np.mean(memory):.2f}MB")
    print(f"Median memory usage: {np.median(memory):.2f}MB")
    print(f"95th percentile memory usage: {np.percentile(memory, 95):.2f}MB")
    print(f"99th percentile memory usage: {np.percentile(memory, 99):.2f}MB")

    
if __name__ == "__main__":
    app()