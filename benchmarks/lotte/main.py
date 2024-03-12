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
    print(collection_dataset[split + '_collection'][0])
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


def colbert_indexing(experiment: str, exp_path: str, dataset: LoTTeDataset, nbits=2, checkpoint: str = "colbert-ir/colbertv2.0"):
     """
     colbert_indexing reads in paths to the collection and queries and indexes the collection using the colbert library.
     """
     with Run().context(RunConfig(nranks=1, experiment=experiment)):
        # config = ColBERTConfig(
        #     nbits=nbits,
        #     kmeans_niters=4,
        #     root=exp_path,
        # )
        config = ColBERTConfig.load_from_checkpoint(checkpoint)
        config.kmeans_niters=4
        # indexer = Indexer(checkpoint=checkpoint, config=config)
        # indexer.index(name=experiment, collection=dataset.collection) # "/path/to/MSMARCO/collection.tsv"

        searcher = Searcher(index=experiment, config=config, collection=dataset.collection)
        # print(searcher.ranker.codec.centroids)
        # print(searcher.ranker.codec.centroids.shape)
        mapped_queries = {id: q for id, q in zip(dataset.qids, dataset.queries)}
        queries = Queries(data = mapped_queries) # "/path/to/MSMARCO/queries.dev.small.tsv"
        ranking = searcher.search_all(queries, k=100)
        ranking.save(f"{experiment}.ranking.tsv")

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

    if not os.path.exists(f"/tmp/py_index_raw_embeddings_{experiment}.npz"):
        # sample and train the index.
        training_data = []
        for doc in tqdm(dataset.collection, desc="embedding training data"):
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
        index = pylintdb.IndexIVF(index_path, 16384, 128, nbits, 4, use_compression)
        
        start = time.perf_counter()
        index.train(dd.astype('float32'))
        train_duration = time.perf_counter() - start
        print(f"Training duration: {train_duration:.2f}s")
        if reuse_centroids:
            # we still train so that the binarizer is trained, but we reuse the centroids
            # from colbert.
            with Run().context(RunConfig(nranks=1, experiment='colbert')):
                searcher = Searcher(index='colbert', config=config, collection=dataset.collection)
                centroids = searcher.ranker.codec.centroids
                index.set_centroids(centroids)
                index.save()

        # add all the documents.
        start = time.perf_counter()
        for i, embedding in tqdm(enumerate(training_data), desc="adding documents"):
            doc = pylintdb.RawPassage(
                embedding,
                i
            )
            index.add([doc])

        index_duration = time.perf_counter() - start
        print(f"Indexing duration: {index_duration:.2f}s")
    else:
        print("Loading index")
        index = pylintdb.IndexIVF(index_path)
        if reuse_centroids:
            # we still train so that the binarizer is trained, but we reuse the centroids
            # from colbert.
            with Run().context(RunConfig(nranks=1, experiment='colbert')):
                searcher = Searcher(index='colbert', config=config, collection=dataset.collection)
                centroids = searcher.ranker.codec.centroids
                index.set_centroids(centroids)
                index.save()
    
    print("Running search")
    with open(f"{exp_path}/{experiment}.ranking.tsv", "w") as f:
        failure_ids=set()
        if failures:
            failure_ids = set(failures.keys())
        for id, query in zip(dataset.qids, dataset.queries):
            if failures and id not in failure_ids:
                continue

            embeddings = checkpoint.queryFromText([query])
            converted = np.squeeze(embeddings[0].numpy().astype('float32'))
            
            expected_pids = failures.get(id, [])

            if expected_pids:
                for pid in expected_pids:
                    print("Searching for pid: ", pid)
                    opts = pylintdb.SearchOptions()
                    opts.expected_id = pid
                    results = index.search(
                        converted, 
                        k, # nprobe
                        100, # k to return
                        opts
                    )
            else:
                results = index.search(
                    converted, 
                    k, # nprobe
                    100, # k to return
                )
            for rank, result in enumerate(results):
                # qid, pid, rank
                f.write(f"{id}\t{result.id}\t{rank+1}\t{result.distance}\n")



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

    success = 0
    success_ids = []
    failure_ids = []
    qas_path = os.path.join(data_path, f"qas.{query_type}.jsonl")

    num_total_qids = 0
    with jsonlines.open(qas_path, mode="r") as f:
        for line in f:
            qid = int(line["qid"])
            if qid not in rankings:
                # print(f"WARNING: qid {qid} not found in {rankings_path}!", file=sys.stderr)
                continue

            num_total_qids += 1
            answer_pids = set(line["answer_pids"])
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
    print(
        "success ids: ", success_ids
    )
    print(
        "failure ids: ", failure_ids
    )


def main():
    dataset = 'lifestyle'
    datasplit = 'dev'
    runtime = 'lintdb' # lintdb

    d = load_lotte(dataset, datasplit, max_id=10000)

    if runtime == 'colbert':
        colbert_indexing('colbert2', '/tmp', d)
        rankings_path = "/home/matt/deployql/LintDB/experiments/colbert/benchmarks.lotte.main/2024-03/04/17.37.19/colbert.ranking.tsv"
    elif runtime == 'lintdb':
        # these are failures when we do our own clustering.
        # If we use the centroids from the colbert model,
        # we only fail on 15, which is parity with colbert.
        failures = {
            5: [5462],
            11: [7767],
            13: [4176, 4185, 5814, 4174],
            15: [1925],
            16: [3701, 3060, 3051, 3437],
            19: [5619]
        }
        # 
        # failures from run without compression. I changed how the  floats are stored and read back.
        # i.e. we didn't cast ebfore, but we are now.
        #failure ids:  [(5, {5462}), (11, {7767}), (13, {4176, 4185, 5814, 4174}), (15, {1925}), (16, {3701, 3050, 3051, 3437}), (19, {5619})]
        experiment = 'colbert'
        lintdb_indexing(
            experiment, 
            'experiments', 
            d, 
            2, 
            nbits=2,
            reuse_centroids=True,
            use_compression=False,
            # failures=failures
        )
        rankings_path = f'experiments/{experiment}.ranking.tsv'

    evaluate_dataset(
        'search', 
        dataset,
        datasplit,
        5,
        'data/lotte/',
        rankings_path
    )

if __name__ == "__main__":
    main()