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
import torch
import random 
from typing import List, Annotated
from common import load_lotte, lintdb_indexing, evaluate_dataset
import tempfile

app = typer.Typer()

@app.command()
def debug():
    torch.set_printoptions(threshold=10_000)

    d = load_lotte('lifestyle', 'dev', filter=True, start=5400, stop=5500)
    print(f"Loaded {len(d.queries)} queries and {len(d.collection):,} passages")
    assert(len(d.collection) == 100)

    with Run().context(RunConfig(nranks=1, experiment='colbert-debug')):
        config = ColBERTConfig.load_from_checkpoint("colbert-ir/colbertv2.0")
        config.kmeans_niters=4
        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)

        doc = None
        for i in range(len(d.collection)):
            if d.dids[i] == 5462:
                doc = d.collection[i]
                break
        if doc is None:
            print("doc not found")
            return
        
        # indexer trains, so needs a larger collection.
        indexer.index(name='colbert-debug', collection=d.collection, overwrite=True)
        # indexer = Indexer(checkpoint=checkpoint, config=config)
        # indexer.index(name=experiment, collection=dataset.collection) # "/path/to/MSMARCO/collection.tsv"
        from colbert.modeling.checkpoint import Checkpoint
        from colbert import Searcher
        searcher = Searcher(index='colbert-debug', config=config, collection=d.collection)

        # spot check this doc

        # doc_len = searcher.ranker.doclens[5462]
        # print(f"doc len: {doc_len}")

        checkpoint = Checkpoint("colbert-ir/colbertv2.0", config)
        doclens_ = checkpoint.docFromText([doc])
        print(f"embedding size: {doclens_.shape}")

        embs_, doclens_ = checkpoint.docFromText([doc],bsize=1,  keep_dims='flatten')
        print(f"embedding size: {doclens_}")

        dddd = searcher.ranker.lookup_pids([5462-5400])
        print(f"shape of searcher's doc: {dddd[0].shape}")

        return

if __name__ == "__main__":
    app()