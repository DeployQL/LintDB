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
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import load_from_dataset

app = typer.Typer()

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


@app.command()
def run(index_path: str = "vidore.index", stop:int=40000, reuse_colbert_clusters=True):
    experiment = ""
    model_name = "vidore/colpali"
    model = ColPali.from_pretrained(
        "google/paligemma-3b-mix-448",
        torch_dtype=torch.bfloat16,
        device_map="cuda").eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # select images -> load_from_pdf(<pdf_path>),  load_from_image_urls(["<url_1>"]), load_from_dataset(<path>)
    images = load_from_dataset("vidore/docvqa_test_subsampled")

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )
    ds = []
    num_embeddings = 0
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
            embs = list(torch.unbind(embeddings_doc.to("cpu")))
            num_embeddings += sum([x.shape(0) for x in embs])
        ds.extend(embs)

    schema = Schema(
        [
            ColbertField('colbert', DataType.TENSOR, {
                'dimensions': 128,
                'quantization': QuantizerType.BINARIZER,
                "num_centroids": np.sqrt(num_embeddings),
                "num_iterations": 10,
            })
        ]
    )
    config = Configuration()
    index = IndexIVF(index_path, schema, config)

    training_docs = []
    for emb in ds:
        emb = np.squeeze(emb.numpy().astype('float32'))
        doc = Document(0, [TensorFieldValue("colbert", emb)])
        training_docs.append(doc)
    index.train(training_docs)

    print("Indexing...")
    start = time.perf_counter()

    latencies = {
        'indexing': [],
    }
    for embedding in ds:
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
