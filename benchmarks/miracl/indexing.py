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

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

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

    index_type_enum = ldb.IndexEncoding_BINARIZER
    if index_type == "binarizer":
        index_type_enum = ldb.IndexEncoding_BINARIZER
    elif index_type == 'pq':
        index_type_enum = ldb.IndexEncoding_PRODUCT_QUANTIZER
    elif index_type == 'none':
        index_type_enum = ldb.IndexEncoding_NONE
    elif index_type == 'xtr':
        index_type_enum = ldb.IndexEncoding_XTR

    print(f"using index type: {index_type_enum}")

        # lifestyle full centroids == 65536
        #lifestyle-40k-benchmark centroids == 32768
    dims = 128
    if index_type_enum == ldb.IndexEncoding_XTR:
        dims = 128
    index = ldb.IndexIVF(index_path, 32768, dims, nbits, 6, 16, index_type_enum)
    opts = ldb.CollectionOptions()
    opts.model_file = "assets/model.onnx"
    opts.tokenizer_file = "assets/colbert_tokenizer.json"
    if index_type_enum == ldb.IndexEncoding_XTR:
        opts.model_file = "assets/xtr/encoder.onnx"
        opts.tokenizer_file = "assets/xtr/spiece.model"
    collection = ldb.Collection(index, opts)

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

if __name__ == "__main__":
    app()