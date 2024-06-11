import lintdb as ldb
from datasets import load_dataset

from tqdm import tqdm
import typer
import random
import time
import os
import pathlib
import csv
import shutil
import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


app = typer.Typer()

# https://github.com/PongoAI/pongo-miracl-benchmark/blob/main/scripts/run-pongo.py
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

@app.command()
def eval(
    experiment: str,
    split: str = 'en',
    use_rerank: bool = True,
):
    dataset = load_dataset('miracl/miracl', split, use_auth_token=True)

    index = ldb.IndexIVF(f"experiments/miracl/{experiment}")
    opts = ldb.CollectionOptions()
    opts.model_file = "/home/matt/deployql/LintDB/assets/model.onnx"
    opts.tokenizer_file = "/home/matt/deployql/LintDB/assets/colbert_tokenizer.json"

    collection = ldb.Collection(index, opts)

    if use_rerank:
        print("loading reranker model...")
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
        model.eval()

    file_exists = False
    try:
        with open('./miracl-lintdb.csv', 'r') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    if not file_exists:
        with open('./miracl-lintdb.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["question", 'relevant_passages', 'lintdb_answer', 'lintdb_mrr', 'lintdb_DCG10', 'iDCG10'])

    with open('./miracl-lintdb.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        for data in tqdm(dataset['dev']):
            question = data['query']

            results = collection.search(0, question, 100)
            if use_rerank:
                print("reranking...")
                texts = [doc.metadata['text'] for doc in results]
                pairs = [(question, text) for text in texts]

                with torch.no_grad():
                    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

                    tups = list(zip(results, scores))
                    results = sorted(tups, key=lambda x: x[1], reverse=True)
                    results = [x[0] for x in results]
                    print("done reranking...")

            mrr = -1
            i = 1
            found = False
            doc_relevances = []
            for doc in results:
                text = doc.metadata['text']
                expected = data['positive_passages']
                docid = doc.metadata['docid']
                for expected_passage in expected:
                    if docid == expected_passage['docid']:
                        if mrr == -1:
                            mrr = i
                        found = True
                        break
                i+=1

            if found:
                doc_relevances.append(1)
            else:
                doc_relevances.append(0)

            DCG10 = 0
            iDCG10 = 0

            # count irrelevant docs and DCG@10
            i = 1
            for rel in doc_relevances:
                DCG10 += rel / math.log2(i + 1)
                i += 1

            # calculate ideal score
            i = 1
            for _ in data['positive_passages']:
                iDCG10 += 1 / math.log2(i + 1)
                i += 1

            writer.writerow([question, '', '', mrr, DCG10, iDCG10])

    #calculate overall MRR@3 and NDCG@10
    with open('./miracl-lintdb.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        mrr3_sum = 0
        mrr5_sum = 0
        DCG10_sum = 0
        iDCG10_sum = 0
        count = 0
        for row in reader:
            mrr3_sum += float(row[3]) if float(row[3]) <=3 else 0
            mrr5_sum += float(row[3]) if float(row[3]) <=5 else 0
            DCG10_sum += float(row[4])
            iDCG10_sum += float(row[5])
            count += 1

        MRR = mrr_sum / count
        NDCG10 = DCG10_sum / iDCG10_sum

        print(f"MRR: {MRR}")
        print(f"NDCG@10: {NDCG10}")



@app.command()
def index(
    experiment: str,
    split: str = 'en',
    k: int = 5, 
    start:int=0, 
    stop:int=40000, 
    num_procs:int=10, 
    nbits: int=1, 
    index_type="binarizer",
    use_batch:bool=False,
    batch_size:int=5,
    checkpoint: str = "colbert-ir/colbertv2.0"):
    print("Loading dataset...")
    dataset = load_dataset('miracl/miracl', split, use_auth_token=True)
    print("Dataset loaded.")

    index_path = f"experiments/miracl/{experiment}"
    # assert not os.path.exists(index_path)
    if index_path and os.path.exists(index_path):
        # delete directory if exists
        shutil.rmtree(index_path)

    index_type_enum = ldb.IndexEncoding_BINARIZER
    if index_type == "binarizer":
        index_type_enum = ldb.IndexEncoding_BINARIZER
    elif index_type == 'pq':
        index_type_enum = ldb.IndexEncoding_PRODUCT_QUANTIZER
    elif index_type == 'none':
        index_type_enum = ldb.IndexEncoding_NONE
    elif index_type == 'xtr':
        index_type_enum = ldb.IndexEncoding_XTR

    print(f"using index type: {index_type_enum}")

        # lifestyle full centroids == 65536
        #lifestyle-40k-benchmark centroids == 32768
    dims = 128

    config = ldb.Configuration()
    config.nbits = nbits
    config.dim = dims
    config.quantizer_type = index_type_enum

    index = ldb.IndexIVF(index_path, config)
    opts = ldb.CollectionOptions()
    opts.model_file = "/home/matt/deployql/LintDB/assets/model.onnx"
    opts.tokenizer_file = "/home/matt/deployql/LintDB/assets/colbert_tokenizer.json"

    collection = ldb.Collection(index, opts)

    id=0
    passages = []
    for data in tqdm(dataset['dev']):
        # query_id = data['query_id']

        passages.extend(data['positive_passages'])
        passages.extend(data['negative_passages'])

    training_data = random.sample(passages, 1000)
    training_data = [x['text'] for x in training_data]

    collection.train(training_data)

    start = time.perf_counter()
    for passage in passages:
        collection.add(0, id, passage['text'], {
            'text': passage['text'],
            'docid': passage['docid'],
            'title': passage['title'],
        })
        id += 1

    duration = time.perf_counter() - start
    print(f"Indexing complete. duration: {duration:.2f}s")

if __name__ == "__main__":
    app()