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


LoTTeDataset = namedtuple('LoTTeDataset', ['collection', 'queries', 'qids', 'dids'])


def load_lotte(dataset, split, filter=True, start=0, stop=500000):
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

def colbert_indexing(experiment: str, exp_path: str, dataset: LoTTeDataset, nbits=2, checkpoint: str = "colbert-ir/colbertv2.0"):
     """
     colbert_indexing reads in paths to the collection and queries and indexes the collection using the colbert library.
     """
     with Run().context(RunConfig(nranks=1, experiment=experiment)):
        config = ColBERTConfig.load_from_checkpoint(checkpoint)
        config.kmeans_niters=4
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=experiment, collection=dataset.collection) # "/path/to/MSMARCO/collection.tsv"

        searcher = Searcher(index=experiment, config=config, collection=dataset.collection)

        mapped_queries = {id: q for id, q in zip(dataset.qids, dataset.queries)}
        queries = Queries(data = mapped_queries) # "/path/to/MSMARCO/queries.dev.small.tsv"
        ranking = searcher.search_all(queries, k=100)
        ranking.save(f"{experiment}.ranking.tsv")

def lintdb_search(
        experiment: str, 
        exp_path: str, 
        dataset:LoTTeDataset, 
        k, 
        nbits=2,  
        checkpoint: str = "colbert-ir/colbertv2.0", 
        reuse_centroids=True, 
        use_compression=False,
        failures={}):
    # let's get the same model.
    checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
    config = ColBERTConfig.from_existing(checkpoint_config, None)

    from colbert.modeling.checkpoint import Checkpoint
    from colbert import Searcher
    checkpoint = Checkpoint(checkpoint, config)

    index_path = f"{exp_path}/py_index_bench_{experiment}"
    if not os.path.exists(index_path):
        print("index not found. exiting")
        return
    else:
        print("Loading index")
        index = ldb.IndexIVF(index_path)
        if reuse_centroids:
            print("the index exists, but we are reusing centroids.",
                  "This isn't supported, because the index relies on the centroids.",
                  "Please delete the index and rerun.")
    
    print("Running search")
    with open(f"{exp_path}/{experiment}.ranking.tsv", "w") as f:
        failure_ids=set()
        if failures:
            failure_ids = set(failures.keys())
        for id, query in zip(dataset.qids, dataset.queries):
            if failures and id not in failure_ids:
                continue
            
            # I want only the query and no padding.
            # obj = checkpoint.query_tokenizer.tok(query, padding=False, truncation=True, return_tensors='pt')
            # ids, mask = obj['input_ids'], obj['attention_mask']
            # embeddings = checkpoint.query(ids, mask)
            embeddings = checkpoint.queryFromText([query])
            converted = np.squeeze(embeddings.numpy().astype('float32'))
            
            expected_pids = failures.get(id, [])

            # it looks like  nprobe should instead of be num tokens * ncells. we use ncells=2.
            k = np.shape(converted)[0] * 2

            if expected_pids:
                print("query id: ", id)
                for pid in expected_pids:
                    print("Searching for pid: ", pid)
                    opts = ldb.SearchOptions()
                    opts.expected_id = pid
                    results = index.search(
                        0, # tenant
                       converted, # converted, 
                        k, # nprobe
                        100, # k to return
                        opts
                    )
            else:
                results = index.search(
                    0,
                    converted, 
                    k, # nprobe
                    100, # k to return
                )
            for rank, result in enumerate(results):
                # qid, pid, rank
                f.write(f"{id}\t{result.id}\t{rank+1}\t{result.score}\n")

def lintdb_indexing(
        experiment: str, 
        exp_path: str, 
        dataset:LoTTeDataset, 
        k, 
        nbits=2,  
        checkpoint: str = "colbert-ir/colbertv2.0", 
        reuse_centroids=True, 
        use_compression=False,
        failures={}):
    # let's get the same model.
    checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
    config = ColBERTConfig.from_existing(checkpoint_config, None)

    from colbert.modeling.checkpoint import Checkpoint
    from colbert import Searcher
    checkpoint = Checkpoint(checkpoint, config)

    if not reuse_centroids:
        if not os.path.exists(f"/tmp/py_index_raw_embeddings_{experiment}.npz"):
            # sample and train the index.
            training_data = []
            sample = random.sample(dataset.collection, 25000)
            for doc in tqdm(sample, desc="embedding training data"):
                embeddings = checkpoint.docFromText([doc])
                training_data.append(np.squeeze(embeddings.numpy().astype('float32')))

            np.savez_compressed(f"/tmp/py_index_raw_embeddings_{experiment}.npz", *training_data)
            dd = np.concatenate(training_data)
        else:
            with open(f"/tmp/py_index_raw_embeddings_{experiment}.npz", "rb") as f:
                arr_maps = np.load(f)
                training_data = [arr for _, arr in arr_maps.items()]
                dd = np.vstack(training_data)

    index_path = f"/tmp/py_index_bench_{experiment}"
    if not os.path.exists(index_path):
        # lifestyle full centroids == 65536
        index = ldb.IndexIVF(index_path, 65536, 128, nbits, 10, use_compression)
        if reuse_centroids:
            # we still train so that the binarizer is trained, but we reuse the centroids
            # from colbert.
            with Run().context(RunConfig(nranks=1, experiment='colbert-lifestyle-full')):
                searcher = Searcher(index='colbert-lifestyle-full', config=config, collection=dataset.collection)
                centroids = searcher.ranker.codec.centroids
                index.set_centroids(centroids)
                index.set_weights(
                    searcher.ranker.codec.bucket_weights.tolist(), 
                    searcher.ranker.codec.bucket_cutoffs.tolist(), 
                    searcher.ranker.codec.avg_residual
                )
                index.save()

        else:
            start = time.perf_counter()
            index.train(dd.astype('float32'))
            train_duration = time.perf_counter() - start
            print(f"Training duration: {train_duration:.2f}s")

        # add all the documents.
        start = time.perf_counter()

        for i, d in tqdm(zip(dataset.dids, dataset.collection), desc="adding documents"):
        # for i, embedding in tqdm(enumerate(training_data), desc="adding documents"):
            embedding = checkpoint.docFromText([d])
            e = embedding[0].numpy().astype('float32')
            doc = ldb.RawPassage(
                e,
                i
            )
            index.add(0, [doc])
                

        index_duration = time.perf_counter() - start
        print(f"Indexing duration: {index_duration:.2f}s")
    else:
        print("Loading index")
        index = ldb.IndexIVF(index_path)
        if reuse_centroids:
            print("the index exists, but we are reusing centroids.",
                  "This isn't supported, because the index relies on the centroids.",
                  "Please delete the index and rerun.")
    
    print("Running search")
    with open(f"{exp_path}/{experiment}.ranking.tsv", "w") as f:
        failure_ids=set()
        if failures:
            failure_ids = set(failures.keys())
        for id, query in zip(dataset.qids, dataset.queries):
            if failures and id not in failure_ids:
                continue
            
            # I want only the query and no padding.
            # obj = checkpoint.query_tokenizer.tok(query, padding=False, truncation=True, return_tensors='pt')
            # ids, mask = obj['input_ids'], obj['attention_mask']
            # embeddings = checkpoint.query(ids, mask)
            embeddings = checkpoint.queryFromText([query])
            converted = np.squeeze(embeddings.numpy().astype('float32'))
            
            expected_pids = failures.get(id, [])

            # it looks like  nprobe should instead of be num tokens * ncells. we use ncells=2.
            k = np.shape(converted)[0] * 2

            if expected_pids:
                print("query id: ", id)
                for pid in expected_pids:
                    print("Searching for pid: ", pid)
                    opts = ldb.SearchOptions()
                    opts.expected_id = pid
                    results = index.search(
                        0, # tenant
                       converted, # converted, 
                        k, # nprobe
                        100, # k to return
                        opts
                    )
            else:
                results = index.search(
                    0,
                    converted, 
                    k, # nprobe
                    100, # k to return
                )
            for rank, result in enumerate(results):
                # qid, pid, rank
                f.write(f"{id}\t{result.id}\t{rank+1}\t{result.score}\n")

def _evaluate_dataset(rankings, dataset:str, query_type: str, split:str='dev', k=5):
    success = 0
    success_ids = []
    failure_ids = []

    queries_dataset = load_dataset("colbertv2/lotte", dataset)
    queries_dataset = queries_dataset['search_'+split]
    # answers = {x['qid']: x['answers']['answer_pids'] for x in queries_dataset}

    num_total_qids = 0
    for line in queries_dataset:
        qid = int(line["qid"])
        if qid not in rankings:
            # print(f"WARNING: qid {qid} not found in {rankings_path}!", file=sys.stderr)
            continue

        num_total_qids += 1
        answer_pids = set(line['answers']["answer_pids"])

        if len(set(rankings[qid][:k]).intersection(answer_pids)) > 0:
            success += 1
            success_ids.append((qid, answer_pids))
        else:
            failure_ids.append((qid, answer_pids))

    print(f"success: {success}, total: {num_total_qids}")
    print(
        f"[query_type={query_type}, dataset={dataset}] "
        f"Success@{k}: {success / num_total_qids * 100:.1f}"
    )
    return success_ids, failure_ids

# copied from colbert/util/evaluate
def evaluate_dataset(query_type, dataset, split, k, data_rootdir, rankings_path):
    data_path = os.path.join(data_rootdir, dataset, split)
    
    if not os.path.exists(rankings_path):
        print("Rankings file not found! Skipping evaluation.")
        return
    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split("\t")
            qid, pid, rank = items[:3]
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            rankings[qid].append(pid)
            assert rank == len(rankings[qid])

    success_ids, failure_ids = _evaluate_dataset(rankings, dataset, split=split, query_type=query_type, k=k)

    with open(f"{rankings_path}.failures", "w") as f:
        for qid, answer_pids in failure_ids:
            f.write(f"{qid}\t{answer_pids}\n")
    print(
        "success ids: ", success_ids
    )
    print(
        "failure ids: ", failure_ids
    )