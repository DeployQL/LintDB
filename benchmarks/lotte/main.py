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

LoTTeDataset = namedtuple('LoTTeDataset', ['collection', 'queries', 'qids'])

def load_lotte(dataset, split, max_id=500000):
    collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
    collection = [x['text'] for x in collection_dataset[split + '_collection']]

    queries_dataset = load_dataset("colbertv2/lotte", dataset)
    queries = [x['query'] for x in queries_dataset['search_' + split]]
    qids = [x['qid'] for x in queries_dataset['search_' + split]]

    f'Loaded {len(queries)} queries and {len(collection):,} passages'

    answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + split]]
    filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(x < max_id for x in apids)]
    filtered_qids = [i for i,(q, apids) in enumerate(zip(queries, answer_pids)) if any(x < max_id for x in apids)]
    f'Filtered down to {len(filtered_queries)} queries'

    return LoTTeDataset(collection[:max_id], filtered_queries, filtered_qids)


def colbert_indexing(experiment: str, exp_path: str, dataset: LoTTeDataset, nbits=2, checkpoint: str = "colbert-ir/colbertv2.0"):
     """
     colbert_indexing reads in paths to the collection and queries and indexes the collection using the colbert library.
     """
     with Run().context(RunConfig(nranks=1, experiment=experiment)):
        config = ColBERTConfig(
            nbits=nbits,
            kmeans_niters=4,
            root=exp_path,
        )
        # indexer = Indexer(checkpoint=checkpoint, config=config)
        # indexer.index(name=experiment, collection=dataset.collection) # "/path/to/MSMARCO/collection.tsv"

        searcher = Searcher(index=experiment, config=config)
        mapped_queries = {id: q for id, q in zip(dataset.qids, dataset.queries)}
        # print(type(mapped_queries))
        queries = Queries(data = mapped_queries) # "/path/to/MSMARCO/queries.dev.small.tsv"
        ranking = searcher.search_all(queries, k=100)
        ranking.save(f"{experiment}.ranking.tsv")

def lintdb_indexing(experiment: str, exp_path: str, dataset:LoTTeDataset, k, nbits=2,  checkpoint: str = "colbert-ir/colbertv2.0", gpu=True):
    # let's get the same model.
    checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
    config = ColBERTConfig.from_existing(checkpoint_config, None)

    from colbert.modeling.checkpoint import Checkpoint
    checkpoint = Checkpoint(checkpoint, config)

    # checkpoint doesn't have .to_gpu() as a method. need to pass this in elsewhere.
    # if gpu:
    #     checkpoint = checkpoint.to_gpu()
    start = time.perf_counter()
    # sample and train the index.
    training_data = []
    for doc in tqdm(dataset.collection, desc="embedding training data"):
        embeddings = checkpoint.docFromText([doc])
        training_data.extend(np.squeeze(embeddings.numpy()))

    index = pylintdb.IndexIVF("/tmp/py_index_bench", 16384, 128, nbits)
    dd = np.stack(training_data)
    index.train(dd)

    train_duration = time.perf_counter() - start
    print(f"Training duration: {train_duration:.2f}s")
    # add all the documents.
    start = time.perf_counter()
    for i, passage in tqdm(enumerate(dataset.collection), desc="adding documents"):
        embeddings = checkpoint.docFromText([passage])
        converted = np.squeeze(embeddings[0].numpy().astype('float32'))
        doc = pylintdb.RawPassage(
            converted,
            i,
            "doc id",
        )
        index.add([doc])

    index_duration = time.perf_counter() - start
    print(f"Indexing duration: {index_duration:.2f}s")

    with open(f"{exp_path}/{experiment}.ranking.tsv", "w") as f:
        for id, query in zip(dataset.qids, dataset.queries):
            results = index.search(query, 100)
            for result in results:
                f.write(f"{query}\t{result}\n")

# copied from colbert/util/evaluate
def evaluate_dataset(query_type, dataset, split, k, data_rootdir, rankings_path):
    data_path = os.path.join(data_rootdir, dataset, split)
    # rankings_path = os.path.join(
    #     rankings_rootdir, split, f"{dataset}.{query_type}.ranking.tsv"
    # )
    
    if not os.path.exists(rankings_path):
        print(f"[query_type={query_type}, dataset={dataset}] Success@{k}: ???")
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
    qas_path = os.path.join(data_path, f"qas.{query_type}.jsonl")

    num_total_qids = 0
    with jsonlines.open(qas_path, mode="r") as f:
        for line in f:
            qid = int(line["qid"])
            if qid not in rankings:
                print(f"WARNING: qid {qid} not found in {rankings_path}!", file=sys.stderr)
                continue

            num_total_qids += 1
            answer_pids = set(line["answer_pids"])
            if len(set(rankings[qid][:k]).intersection(answer_pids)) > 0:
                success += 1

    print(f"success: {success}, total: {num_total_qids}")
    print(
        f"[query_type={query_type}, dataset={dataset}] "
        f"Success@{k}: {success / num_total_qids * 100:.1f}"
    )


def main():
    dataset = 'lifestyle'
    datasplit = 'dev'
    runtime = 'lintdb' # lintdb
    k = 5

    d = load_lotte(dataset, datasplit, max_id=10000)

    if runtime == 'colbert':
        colbert_indexing('colbert', '/tmp', d)
        rankings_path = "/home/matt/deployql/LintDB/experiments/colbert/benchmarks.lotte.main/2024-03/04/17.37.19/colbert.ranking.tsv"
    elif runtime == 'lintdb':
        lintdb_indexing('colbert', '/tmp', d, 5, nbits=2)
        rankings_path = '/tmp/colbert.ranking.tsv'

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