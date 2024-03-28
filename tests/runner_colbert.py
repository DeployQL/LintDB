import lintdb as ldb
from datasets import load_dataset
from collections import namedtuple
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert.modeling.checkpoint import Checkpoint
from colbert.index_update import IndexUpdater
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
from lotte.common import load_lotte
from common import get_memory_usage


def single_search(experiment='colbert-lifestyle-full', dataset:str='lifestyle', split:str='dev', checkpoint:str='colbert-ir/colbertv2.0', index_path:str='indexes/lifestyle'):
    d = load_lotte(dataset, split, stop=40000000)
    latencies = []
    memory = []

    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        config = ColBERTConfig.load_from_checkpoint(checkpoint)
        config.kmeans_niters=4
        config.ncells = 2
        model = Checkpoint(checkpoint, config)

        updater = IndexUpdater(config, searcher, checkpoint)

        # indexer = Indexer(checkpoint=checkpoint, config=config)
        # indexer.index(name=experiment, collection=dataset.collection) # "/path/to/MSMARCO/collection.tsv"

        searcher = Searcher(index=experiment, config=config, collection=d.collection)

        for query in d.queries:
            embeddings = searcher.encode([query])

            # embeddings = embeddings.squeeze()
            # print(embeddings.shape)

            start = time.perf_counter()
            # searcher.dense_search(embeddings, k=100)
            results = searcher._search_all_Q(Queries.cast({1: query}), embeddings, k=100)
            latencies.append(time.perf_counter() - start)
            memory.append(get_memory_usage())

    
if __name__ == "__main__":
    single_search()