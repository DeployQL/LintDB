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
from common import load_lotte, lintdb_indexing, evaluate_dataset, lintdb_search

app = typer.Typer()


"""
This command only searches a prebuilt index. Use lotte/multiprocess_indexing.py to index
"""
@app.command()
def lintdb(dataset, experiment, split='dev', k=5, checkpoint: str = "colbert-ir/colbertv2.0"):
    d = load_lotte(dataset, split, stop=40000)

    lintdb_search(
        experiment, 
        'experiments', 
        d, 
        checkpoint=checkpoint,
        reuse_centroids=True,
        use_compression=True
    )

    evaluate_dataset(
        'search', 
        dataset,
        split,
        int(k),
        'data/lotte/',
        f'experiments/{experiment}.ranking.tsv'
    )

def comma_separated(raw: str) -> List[int]:
    if isinstance(raw, list):
        return raw
    return [int(x) for x in raw.split(",")]

@app.command()
def run_failures(dataset, experiment, split='dev',  failure: Annotated[list, typer.Option(parser=comma_separated)] = [], checkpoint: str = "colbert-ir/colbertv2.0"):
    d = load_lotte(dataset, split, stop=40000)
    with open(f"experiments/{experiment}.ranking.tsv.failures", "r") as f:
        failures = {}
        for line in f:
            qid, apids = line.strip().split("\t")
            apids = apids.replace("{", "").replace("}", "")
            if (int(qid) in failure) or not failure:
                failures[int(qid)] = [int(x) for x in apids.split(",")]

    failures = {
        # 0: [2466, 2435, 1641, 4619, 1615]
        # 5: [5462],
        # 11: [7767],
        # 13: [4176, 4185, 5814, 4174],
        # 15: [1925],
        # 16: [3701, 3060, 3051, 3437],
        16: [3701]
        # 19: [5619]
    }
    lintdb_search(
        experiment, 
        'experiments', 
        d, 
        2, 
        nbits=2,
        checkpoint=checkpoint,
        reuse_centroids=True,
        use_compression=True,
        failures=failures
    )

@app.command()
def run_failures_colbert(dataset, experiment, split='dev', k:int=5, failure: Annotated[list, typer.Option(parser=comma_separated)] = [], checkpoint: str = "colbert-ir/colbertv2.0"):
    d = load_lotte(dataset, split, stop=40000000)


    # these are failures from lintdb
    # with open(f"experiments/{experiment}.ranking.tsv.failures", "r") as f:
    #     failures = {}
    #     for line in f:
    #         qid, apids = line.strip().split("\t")
    #         apids = apids.replace("{", "").replace("}", "")
    #         failures[int(qid)] = [int(x) for x in apids.split(",")]

    failures = {
        # 0: [2466, 2435, 1641, 4619, 1615]
        # 5: [5462],
        # 11: [7767],
        # 13: [4176, 4185, 5814, 4174],
        # 15: [1925],
        # 16: [3701, 3060, 3051, 3437],
        16: [3701]
        # 19: [5619]
    }

    with Run().context(RunConfig(nranks=1, experiment='colbert-lifestyle-40k-benchmark')):
        config = ColBERTConfig.load_from_checkpoint(checkpoint)
        config.kmeans_niters=4
        config.ncells = 2
        config.ndocs=1024
        config.centroid_score_threshold=.45
        # indexer = Indexer(checkpoint=checkpoint, config=config)
        # indexer.index(name=experiment, collection=dataset.collection) # "/path/to/MSMARCO/collection.tsv"
        from colbert.modeling.checkpoint import Checkpoint
        from colbert import Searcher
        searcher = Searcher(index='colbert-lifestyle-40k-benchmark', config=config, collection=d.collection)
        
        failure = failure if failure else list(failures.keys())
        for id, apids in [(int(x), failures.get(int(x), [])) for x in failure]:
            print("query id: ", id)
            text = d.queries[id]
            # searcher.search(query, k=100)
            # ranker.rank
            # -> ranker.retrieve
            # -> -> generate_candidates
            # -> -> -> get cells
            Q = searcher.encode(text, full_length_search=False)
            print(Q.shape)
            Q_ = Q[:,:32]
            print(Q_.shape)
            Q_ = Q_.squeeze(0)
            cells, scores = searcher.ranker.get_cells(Q_, 2)
            print("num cells: ", len(cells.tolist()))
            print("cells: ", cells)
            print("scores: ", scores)

            # top = scores.topk(2, dim=0, sorted=False)
            # print("top indices: ", top.indices.tolist())
            # print("top scores: ", top.values.tolist())

            # doing this results in slightly different scores than if I call retrieve directly?
            # pids, scores = searcher.ranker.generate_candidate_pids(Q_, 2)
            # print("num pids non-unique: ", len(pids.tolist()))
            # import torch
            # sorter = pids.sort()
            # pids = sorter.values
            # pids, pids_count = torch.unique_consecutive(pids, return_counts=True)
            # print("num pids unqiue: ", len(pids.tolist()))

            # this is part of searcher.ranker.rank
            pids, centroid_scores = searcher.ranker.retrieve(config, Q)
            idx = centroid_scores.max(-1).values >= config.centroid_score_threshold
            # get the filtered pids
            # this filtered list isn't the same as the one from rank() ???
            # filtered_pids = searcher.ranker.filter_pids(pids, centroid_scores, searcher.ranker.embeddings.codes, searcher.ranker.doclens, searcher.ranker.offsets, idx, 1024)
            # print("num filtered pids: ", len(filtered_pids.tolist()))
            # for pid in apids:
            #     try:
            #         index = filtered_pids.tolist().index(pid)
            #         print(f"filtered pid: {pid} found at index: {index}.")
            #     except:
            #         print("pid not found in filtered pid list: ", pid)

            # this gives us the final scores.
            # scores, pids = searcher.ranker.score_pids(config, Q, pids, centroid_scores)
            pids, scores = searcher.ranker.rank(config, Q)
            # print("pids: ", pids)
            for pid in apids:
                try:
                    index = pids.index(pid)
                    print(f"pid: {pid} found at index: {index}. score: {scores[index]}")
                    if index <= k:
                        print("pid found in top k")
                except:
                    print("pid not found: ", pid)
                    continue

if __name__ == "__main__":
    app()
