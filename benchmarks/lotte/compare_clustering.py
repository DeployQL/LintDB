from pylintdb import pylintdb
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

LoTTeDataset = namedtuple('LoTTeDataset', ['collection', 'queries', 'qids', 'dids'])

def load_lotte(dataset, split, max_id=500000):
    collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
    collection = [x['text'] for x in collection_dataset[split + '_collection']]
    dids = [x['doc_id'] for x in collection_dataset[split + '_collection']]

    queries_dataset = load_dataset("colbertv2/lotte", dataset)
    queries = [x['query'] for x in queries_dataset['search_' + split]]
    qids = [x['qid'] for x in queries_dataset['search_' + split]]

    f'Loaded {len(queries)} queries and {len(collection):,} passages'

    answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + split]]
    filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(x < max_id for x in apids)]
    filtered_qids = [i for i,(q, apids) in enumerate(zip(queries, answer_pids)) if any(x < max_id for x in apids)]
    filtered_dids = [x for x in dids if x < max_id]
    f'Filtered down to {len(filtered_queries)} queries'

    return LoTTeDataset(collection[:max_id], filtered_queries, filtered_qids, filtered_dids)

def compare_clustering(experiment, lintdb_path, data):
    from colbert.modeling.checkpoint import Checkpoint
    from colbert import Searcher

    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        # config = ColBERTConfig(
        #     nbits=nbits,
        #     kmeans_niters=4,
        #     root=exp_path,
        # )
        checkpoint_config = ColBERTConfig.load_from_checkpoint("colbert-ir/colbertv2.0")
        config = ColBERTConfig.from_existing(checkpoint_config, None)

        from colbert.modeling.checkpoint import Checkpoint
        from colbert import Searcher
        # checkpoint = Checkpoint("colbert-ir/colbertv2.0", config)

        searcher = Searcher(index=experiment, config=config, collection=data.collection)

        index = pylintdb.IndexIVF(lintdb_path)

        for i in range(16384):
            pids, cell_lengths = searcher.ranker.ivf.lookup([i])

            lintdb_pids = index.lookup_pids(i)

            diff = set([x.item() for x in pids]) - set(lintdb_pids)
            if diff:
                print(
                    f"centroid {i} comparison:",
                    f"colbert: {len(pids)}",
                    f"lintdb: {len(lintdb_pids)}",
                    f"difference: {len(diff)}",
                    f"pid difference: {diff}",  
                )
                for pid_values in failures.values():
                    for pid in pid_values:
                        if pid in diff:
                            print(f"centroid {i} has a failure at pid {pid}")



if __name__ == '__main__':
    dataset = 'lifestyle'
    datasplit = 'dev'

    experiment = 'colbert'

    failures = {
        5: [5462],
        11: [7767],
        13: [4176, 4185, 5814, 4174],
        15: [1925],
        16: [3701, 3060, 3051, 3437],
        19: [5619]
    }

    data = load_lotte(dataset, datasplit)

    compare_clustering(experiment, f"/tmp/py_index_bench_{experiment}", data)