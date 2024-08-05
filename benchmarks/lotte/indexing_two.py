from lintdb.core import (
    Schema,
    ColbertField,
    StoredField,
    IndexedField,
    ContextField,
    QuantizerType,
    Binarizer,
    Configuration,
    FaissCoarseQuantizer,
    SearchOptions,
    DataType,
    FieldValue,
    TensorFieldValue,
    QuantizedTensorFieldValue,
    IntFieldValue,
    FloatFieldValue,
    TextFieldValue,
    DateFieldValue,
    Document,
    IndexIVF,
    Version,
    TermQueryNode,
    VectorQueryNode,
    AndQueryNode,
    Query
)
from colbert.infra import Run, RunConfig, ColBERTConfig
from tqdm import tqdm
import time
import shutil
import numpy as np
import typer
from rich import print
import random
import torch

from common import load_lotte, lintdb_indexing, evaluate_dataset


app = typer.Typer()

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


@app.command()
def run(index_path: str = "local_db.index", stop:int=40000, reuse_colbert_clusters=True):
    experiment = ""
    checkpoint = "colbert-ir/colbertv2.0"
    dataset = 'lifestyle'
    split = 'dev'

    print("Beginning indexing...")

    from colbert.modeling.checkpoint import Checkpoint
    from colbert import Searcher
    config = ColBERTConfig.load_from_checkpoint(checkpoint)
    checkpoint = Checkpoint(checkpoint, config)

    schema = Schema(
        [
            ColbertField('colbert', DataType.TENSOR, {
                'dimensions': 128,
                'quantization': QuantizerType.BINARIZER,
                "num_centroids": 32768,
                "num_iterations": 10,
            })
        ]
    )
    config = Configuration()
    index = IndexIVF(index_path, schema, config)

    d = load_lotte(dataset, split, stop=stop)
    # lifestyle full centroids == 65536
    #lifestyle-40k-benchmark centroids == 32768

    if not reuse_colbert_clusters:
        print("Training...")
        training_docs = []
        training_data = random.sample(d.collection, min(20000, len(d.collection)))
        for b in tqdm(batch(training_data, n=1000)):
            embeddings = checkpoint.docFromText(b)
            for emb in embeddings:
                emb = np.squeeze(emb.cpu().numpy().astype('float32'))
                doc = Document(0, [TensorFieldValue("colbert", emb)])
                training_docs.append(doc)

        index.train(training_docs)
    else:
        print("Reusing colbert centroids")
        from colbert import Searcher
        from colbert.infra import Run, RunConfig
        with Run().context(RunConfig(nranks=1, experiment='colbert')):
            searcher = Searcher(index='colbert', collection=d.collection)
            centroids = searcher.ranker.codec.centroids.cpu().numpy().astype('float32')

            coarse_quantizer = FaissCoarseQuantizer(centroids)
            index.set_coarse_quantizer('colbert', coarse_quantizer)

            binarizer = Binarizer(
                searcher.ranker.codec.bucket_cutoffs.tolist(),
                searcher.ranker.codec.bucket_weights.tolist(),
                searcher.ranker.codec.avg_residual,
                1,
                128
            )
            index.set_quantizer('colbert', binarizer)
            index.save()

    print("Indexing...")
    start = time.perf_counter()

    latencies = {
        'embedding': [],
        'indexing': [],
        'per_doc': [],
    }
    for b in tqdm(batch(list(zip(d.dids, d.collection)),n=1)):
        ids = [i for i,_ in b]
        docs = [d for _, d in b]

        start = time.perf_counter()
        embedding = checkpoint.docFromText(docs)
        end = time.perf_counter()
        latencies['embedding'].append(end - start)

        start = time.perf_counter()
        e = np.squeeze(embedding.cpu().numpy().astype('float32'))

        for i, ee in zip(ids, e):
            start = time.perf_counter()
            doc = Document(i, [TensorFieldValue("colbert", e)])
            index.add(0, [doc])
            end = time.perf_counter()
            latencies['per_doc'].append(end - start)

        latencies['indexing'].append(time.perf_counter() - start)

    duration = time.perf_counter() - start
    print(f"Indexing complete. duration: {duration:.2f}s")
    print('latencies:')
    print(f"p95 embedding latency: {np.percentile(latencies['embedding'], 95):.2f}s")
    print(f"p95 indexing latency: {np.percentile(latencies['indexing'], 95):.2f}s")
    print(f"p95 per doc indexing latency: {np.percentile(latencies['per_doc'], 95):.2f}s")


@app.command()
def eval(index_path = "local_db_2.index", dataset: str = 'lifestyle', split: str = 'dev', stop: int = 40000):
    checkpoint = "colbert-ir/colbertv2.0"
    experiment=""

    from colbert.modeling.checkpoint import Checkpoint
    from colbert import Searcher
    config = ColBERTConfig.load_from_checkpoint(checkpoint)
    checkpoint = Checkpoint(checkpoint, config)
    index = IndexIVF(index_path)

    data = load_lotte(dataset, split, stop=int(stop))

    with open(f"experiments/{experiment}.ranking.tsv", "w") as f:
        for id, query in zip(data.qids, data.queries):
            embeddings = checkpoint.queryFromText([query], bsize=1)
            normalized = torch.nn.functional.normalize(embeddings, p=2, dim=2)
            converted = np.squeeze(normalized.cpu().numpy().astype('float32'))

            query = Query(
                VectorQueryNode(
                    TensorFieldValue('colbert', converted)
                )
            )
            results = index.search(
                0,
                query,
                100,
                {
                    'k_top_centroids': 32,
                }
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

if __name__ == "__main__":
    app()
