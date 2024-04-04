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
from common import load_lotte, lintdb_indexing, evaluate_dataset
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from colbert.modeling.checkpoint import Checkpoint
from colbert import Searcher

app = typer.Typer()

model = None

def intialize_model(checkpoint):
    checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
    config = ColBERTConfig.from_existing(checkpoint_config, None)

    global model
    model = Checkpoint(checkpoint, config)

def encode_one(doc_tuple):
    id, text = doc_tuple
    embedding = model.docFromText([text], bsize=1, keep_dims='flatten')
    e = embedding[0].numpy().astype('float32')
    return (id, e)

def encode_task(queue, result_queue, dataset, experiment, split, checkpoint):
    checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
    config = ColBERTConfig.from_existing(checkpoint_config, None)

    from colbert.modeling.checkpoint import Checkpoint
    from colbert import Searcher
    checkpoint = Checkpoint(checkpoint, config)

    while True:
        doc_tuple = queue.get()
        if doc_tuple is None:
            return
        
        id, text = doc_tuple
        embedding = checkpoint.docFromText([text])
        e = embedding[0].numpy().astype('float32')
        doc = (id, e)
        result_queue.put(doc)


def consume_task(result_queue, experiment, nbits, use_compression, checkpoint):
    assert not os.path.exists(index_path)
    checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
    config = ColBERTConfig.from_existing(checkpoint_config, None)

    index_path = f"experiments/py_index_bench_{experiment}"
        # lifestyle full centroids == 65536
    index = ldb.IndexIVF(index_path, 65536, 128, nbits, 10, use_compression)
    # in multiprocessing, we only allow for reuse of centroids.
    with Run().context(RunConfig(nranks=1, experiment='colbert-lifestyle-full')):
        searcher = Searcher(index='colbert-lifestyle-full', config=config)
        centroids = searcher.ranker.codec.centroids
        index.set_centroids(centroids)
        index.set_weights(
            searcher.ranker.codec.bucket_weights.tolist(), 
            searcher.ranker.codec.bucket_cutoffs.tolist(), 
            searcher.ranker.codec.avg_residual
        )
        index.save()

    while True:
        result = result_queue.get()
        if result is None:
            break
        
        id, embed = result
        if id % 1000 == 0:
            print(f"Indexing {id}")
        doc = ldb.RawPassage(embed, id)
        index.add(0, [doc])

@app.command()
def run(dataset: str, experiment: str, split: str = 'dev', k: int = 5, start:int=0, stop:int=40000, num_procs:int=10, nbits: int=1, use_compression: bool = True, checkpoint: str = "colbert-ir/colbertv2.0"):
    print("Loading dataset...")
    d = load_lotte(dataset, split, stop=40000)
    print("Dataset loaded.")

    index_path = f"experiments/py_index_bench_{experiment}"
    assert not os.path.exists(index_path)
        # lifestyle full centroids == 65536
        #lifestyle-40k-benchmark centroids == 32768
    index = ldb.IndexIVF(index_path, 32768, 128, nbits, 4, use_compression)
    # in multiprocessing, we only allow for reuse of centroids.
    with Run().context(RunConfig(nranks=1, experiment='colbert-lifestyle-40k-benchmark')):
        checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
        config = ColBERTConfig.from_existing(checkpoint_config, None)
        searcher = Searcher(index='colbert-lifestyle-40k-benchmark', config=config, collection=d.collection)
        centroids = searcher.ranker.codec.centroids
        index.set_centroids(centroids)
        index.set_weights(
            searcher.ranker.codec.bucket_weights.tolist(), 
            searcher.ranker.codec.bucket_cutoffs.tolist(), 
            searcher.ranker.codec.avg_residual
        )
        index.save()

    start = time.perf_counter()

    pool = mp.Pool(processes=num_procs, initializer=intialize_model, initargs=(checkpoint,))

    def create_tuples():
        for i, dd in zip(d.dids, d.collection):
            yield (i, dd)

    for id, embedding in tqdm(pool.imap_unordered(encode_one, create_tuples()), total=270000):
        doc = ldb.RawPassage(embedding, id)
        index.add(0, [doc])

    duration = time.perf_counter() - start
    print(f"Indexing complete. duration: {duration:.2f}s")

if __name__ == "__main__":
    app()