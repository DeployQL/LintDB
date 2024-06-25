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
import shutil
import numpy as np
import typer
import random
import torch
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

def create_collection(index_path, index_type, dims, nbits, num_subquantizers=64, num_centroids=32768):
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
    stop:int=1000,
    num_procs:int=10, 
    nbits: int=1, 
    index_type="binarizer",
    use_batch:bool=False,
    batch_size:int=5,
    checkpoint: str = "colbert-ir/colbertv2.0",
    use_collection:bool=False,
    reuse_centroids:bool=True,
    ):
    print("Loading dataset...")
    d = load_lotte(dataset, split, stop=stop)
    print("Dataset loaded.")

    index_path = f"experiments/py_index_bench_{experiment}"
    if os.path.exists(index_path):
        shutil.rmtree(index_path)

    index_type_enum = get_index_type(index_type)

    # lifestyle full centroids == 65536
    #lifestyle-40k-benchmark centroids == 32768
    if use_collection:
        index, collection = create_collection(index_path, index_type_enum, 128, nbits, num_centroids=32768)
    else:
        from colbert.modeling.checkpoint import Checkpoint
        from colbert import Searcher
        config = ColBERTConfig.load_from_checkpoint(checkpoint)
        checkpoint = Checkpoint(checkpoint, config)
        index = ldb.IndexIVF(index_path, 4096, 128, nbits, 6, 16, index_type_enum)

    training_data = random.sample(d.collection, min(20000, len(d.collection)))

    if use_collection:
        collection.train(training_data, 4096, 10)
    else:
        if reuse_centroids:
            # we still train so that the binarizer is trained, but we reuse the centroids
            # from colbert.
            with Run().context(RunConfig(nranks=1, experiment='colbert-202400622')):
                searcher = Searcher(index='colbert-202400622', config=config, collection=d.collection)
                centroids = searcher.ranker.codec.centroids.cpu().numpy().astype('float32')
                index.set_centroids(centroids)
                index.set_weights(
                    searcher.ranker.codec.bucket_weights.tolist(),
                    searcher.ranker.codec.bucket_cutoffs.tolist(),
                    searcher.ranker.codec.avg_residual
                )
                index.save()
        else:
            training_embs = []
            for doc in tqdm(training_data, desc="embedding training data"):
                embeddings = checkpoint.docFromText([doc])
                training_embs.append(np.squeeze(embeddings.numpy().astype('float32')))
            index.train(training_embs)

    start = time.perf_counter()

    for b in tqdm(batch(list(zip(d.dids, d.collection)),n=25)):
        if use_collection:
            for i, document in b:
                collection.add(0, i, document)
        else:
            ids = [i for i,_ in b]
            docs = [d for _, d in b]
            embedding = checkpoint.docFromText(docs)
            e = embedding.cpu().numpy().astype('float32')
            for i, ee in zip(ids, e):
                index.add(0, [{'id': i, 'embeddings': ee, 'metadata': {}}])




    duration = time.perf_counter() - start
    print(f"Indexing complete. duration: {duration:.2f}s")

@app.command()
def eval(dataset, experiment, index_type='binarizer', split='dev', stop=1000,checkpoint: str = "colbert-ir/colbertv2.0", use_collection:bool=False):
    index_type_enum = get_index_type(index_type)

    if use_collection:
        index, collection = open_collection(f"experiments/py_index_bench_{experiment}", index_type_enum)
    else:
        from colbert.modeling.checkpoint import Checkpoint
        from colbert import Searcher
        config = ColBERTConfig.load_from_checkpoint(checkpoint)
        checkpoint = Checkpoint(checkpoint, config)
        index = ldb.IndexIVF(f"experiments/py_index_bench_{experiment}")

    data = load_lotte(dataset, split, stop=int(stop))

    with Run().context(RunConfig(nranks=1, experiment='colbert-202400622')):
        searcher = Searcher(index='colbert-202400622', config=config, collection=data.collection)
        centroids = searcher.ranker.codec.centroids.cpu().numpy().astype('float32')
        print(searcher.ranker.codec.bucket_cutoffs.tolist())
        index.set_centroids(centroids)
        index.set_weights(
            searcher.ranker.codec.bucket_weights.tolist(),
            searcher.ranker.codec.bucket_cutoffs.tolist(),
            searcher.ranker.codec.avg_residual
        )
        index.save()



    with open(f"experiments/{experiment}.ranking.tsv", "w") as f:
        for id, query in zip(data.qids, data.queries):
            opts = ldb.SearchOptions()
            opts.k_top_centroids = 32
            opts.expected_id = 509
            # opts.nearest_tokens_to_fetch = 100
            if use_collection:
                results = collection.search(
                    0, # tenant
                    query, # converted,
                    100, # k to return
                    opts
                )
            else:
                embeddings = checkpoint.queryFromText([query], bsize=1)
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=2)
                converted = np.squeeze(normalized.cpu().numpy().astype('float32'))
                results = index.search(
                    0,
                    converted,
                    100,
                    opts
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
def run_failures(dataset, experiment, split='dev', index_type="binarizer", use_collection:bool=False, checkpoint: str = "colbert-ir/colbertv2.0"):
    index_type_enum = get_index_type(index_type)
    if use_collection:
        index, collection = open_collection(f"experiments/py_index_bench_{experiment}", index_type_enum)
    else:
        from colbert.modeling.checkpoint import Checkpoint
        from colbert import Searcher
        config = ColBERTConfig.load_from_checkpoint(checkpoint)
        checkpoint = Checkpoint(checkpoint, config)
        index = ldb.IndexIVF(f"experiments/py_index_bench_{experiment}")

    # it's been easier to just hardcode the failures
    failures = {
        # 0: [2466, 2435, 1641, 4619, 1615]
        # 5: [5462],
        # 2: [3457, 8406, 2837, 2838],
        # 11: [7767],
        # 13: [4176, 4185, 5814, 4174],
        # 15: [1925],
        16: [3701],
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
            if use_collection:
                results = collection.search(
                    0, # tenant
                    query, # string query,
                    100, # k to return
                    opts
                )
            else:
                embeddings = checkpoint.queryFromText([query], bsize=1)
                print("embedding shape: ", embeddings.shape())
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=2)
                converted = np.squeeze(normalized.cpu().numpy().astype('float32'))
                results = index.search(
                    0,
                    converted,
                    100,
                    opts
                )



if __name__ == "__main__":
    app()