import lintdb as ldb
from lintdb import (IndexEncoding_BINARIZER, IndexEncoding_NONE, IndexEncoding_PRODUCT_QUANTIZER)
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
from common import load_lotte, lintdb_indexing, evaluate_dataset
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from colbert.modeling.checkpoint import Checkpoint
from colbert import Searcher

app = typer.Typer()

model_files = {
    ldb.IndexEncoding_XTR: {
        'model_file': "assets/xtr/encoder.onnx",
        'tokenizer_file': "assets/xtr/spiece.model",
    },
    ldb.IndexEncoding_BINARIZER: {
        'model_file': "assets/model.onnx",
        'tokenizer_file': "assets/colbert_tokenizer.json",
    },
}

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def open_collection(index_path, index_type):
    index = ldb.IndexIVF(index_path)
    opts = ldb.CollectionOptions()
    opts.model_file = model_files[index_type]['model_file']
    opts.tokenizer_file = model_files[index_type]['tokenizer_file']

    collection = ldb.Collection(index, opts)

    return index, collection

def create_collection(index_path, index_type, dims, nbits, num_subquantizers=16, num_centroids=32768):
    index = ldb.IndexIVF(index_path, num_centroids, dims, nbits, 6, num_subquantizers, index_type)
    opts = ldb.CollectionOptions()
    opts.model_file = model_files[index_type]['model_file']
    opts.tokenizer_file = model_files[index_type]['tokenizer_file']

    collection = ldb.Collection(index, opts)

    return index, collection

def get_index_type(index_type):
    index_type_enum = ldb.IndexEncoding_BINARIZER
    if index_type == "binarizer":
        index_type_enum = ldb.IndexEncoding_BINARIZER
    elif index_type == 'pq':
        index_type_enum = ldb.IndexEncoding_PRODUCT_QUANTIZER
    elif index_type == 'none':
        index_type_enum = ldb.IndexEncoding_NONE
    elif index_type == 'xtr':
        index_type_enum = ldb.IndexEncoding_XTR

    return index_type_enum

@app.command()
def run(
    dataset: str, 
    experiment: str, 
    split: str = 'dev', 
    k: int = 5, 
    start:int=0, 
    stop:int=40000, 
    num_procs:int=10, 
    nbits: int=1, 
    index_type="binarizer",
    use_batch:bool=False,
    batch_size:int=5,
    checkpoint: str = "colbert-ir/colbertv2.0"):
    print("Loading dataset...")
    d = load_lotte(dataset, split, stop=40000)
    print("Dataset loaded.")

    index_path = f"experiments/py_index_bench_{experiment}"
    assert not os.path.exists(index_path)

    index_type_enum = get_index_type(index_type)

    # lifestyle full centroids == 65536
    #lifestyle-40k-benchmark centroids == 32768
    index, collection = create_collection(index_path, index_type_enum, 128, nbits, num_centroids=32768)

    training_data = random.sample(d.collection, 1000)
    collection.train(training_data)

    start = time.perf_counter()

    if use_batch:
        for b in tqdm(batch(list(zip(d.dids, d.collection)), n=batch_size)):
                bb = [{'id': i, 'text': dd, 'metadata': {'text': dd}} for i, dd in b]
                collection.add_batch(0, bb)
    else:
        for i, dd in tqdm(zip(d.dids, d.collection)):
            collection.add(0, i, dd, {'text': dd})

    duration = time.perf_counter() - start
    print(f"Indexing complete. duration: {duration:.2f}s")

@app.command()
def eval(dataset, experiment, index_type='binarizer', split='dev'):
    index_type_enum = get_index_type(index_type)

    index, collection = open_collection(f"experiments/py_index_bench_{experiment}", index_type_enum)
    data = load_lotte(dataset, split, stop=40000)

    with open(f"experiments/{experiment}.ranking.tsv", "w") as f:
        for id, query in zip(data.qids, data.queries):
            opts = ldb.SearchOptions()
            opts.k_top_centroids = 32
            results = collection.search(
                0, # tenant
                query, # converted,
                100, # k to return
                opts,
            )
            for rank, result in enumerate(results):
                # qid, pid, rank
                f.write(f"{id}\t{result.id}\t{rank+1}\t{result.score}\n")

    evaluate_dataset(
        'search',
        dataset,
        split,
        5, # evaluate (e.v.) mrr@5
        'data/lotte/',
        f'experiments/{experiment}.ranking.tsv'
    )

@app.command()
def run_failures(dataset, experiment, split='dev', index_type="binarizer"):
    index_type_enum = get_index_type(index_type)

    index, collection = open_collection(f"experiments/py_index_bench_{experiment}", index_type_enum)

    # it's been easier to just hardcode the failures
    failures = {
        # 0: [2466, 2435, 1641, 4619, 1615]
        # 5: [5462],
        2: [3457, 8406, 2837, 2838],
        # 11: [7767],
        # 13: [4176, 4185, 5814, 4174],
        # 15: [1925],
        # 16: [3701, 3060, 3051, 3437],
        # 16: [3701]
        # 19: [5619]
    }

    dataset = load_lotte(dataset, split, stop=40000)

    failure_ids=set()
    if failures:
        failure_ids = set(failures.keys())

    for id, query in zip(dataset.qids, dataset.queries):
        if failures and id not in failure_ids:
            continue

        expected_pids = failures.get(id, [])

        print("query id: ", id)
        for pid in expected_pids:
            print("Searching for pid: ", pid)
            opts = ldb.SearchOptions()
            opts.expected_id = pid
            opts.k_top_centroids = 32768
            results = collection.search(
                0, # tenant
                query, # string query,
                100, # k to return
                opts
            )



if __name__ == "__main__":
    app()