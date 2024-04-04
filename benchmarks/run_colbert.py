from datasets import load_dataset
from collections import namedtuple
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert.modeling.checkpoint import Checkpoint
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
from lotte.common import load_lotte, _evaluate_dataset
from common import get_memory_usage

app = typer.Typer()


@app.command()
def single_search(experiment='colbert-lifestyle-40k-benchmark', dataset:str='lifestyle', split:str='dev', checkpoint:str='colbert-ir/colbertv2.0', index_path:str='indexes/lifestyle'):
    d = load_lotte(dataset, split, stop=40000)
    latencies = []
    memory = []

    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        config = ColBERTConfig.load_from_checkpoint(checkpoint)
        config.kmeans_niters=4
        config.ncells = 2
        # model = Checkpoint(checkpoint, config)

        # indexer = Indexer(checkpoint=checkpoint, config=config)
        # indexer.index(name=experiment, collection=dataset.collection) # "/path/to/MSMARCO/collection.tsv"

        searcher = Searcher(index=experiment, config=config, collection=d.collection)
        rankings = {}

        for id, query in zip(d.qids, d.queries):
            embeddings = searcher.encode([query])

            start = time.perf_counter()
            results = searcher._search_all_Q(Queries.cast({1: query}), embeddings, k=100)
            latencies.append(time.perf_counter() - start)
            memory.append(get_memory_usage())

            for k, v in results.todict().items():
                rankings[id] = [x[0] for x in v]

        _evaluate_dataset(rankings, dataset, 'search', k=5)


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