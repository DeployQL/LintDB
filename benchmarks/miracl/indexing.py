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
import numpy as np

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
    index = ldb.IndexIVF(index_path, num_centroids, dims, nbits, 10, num_subquantizers, index_type)
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

# https://github.com/PongoAI/pongo-miracl-benchmark/blob/main/scripts/run-pongo.py
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

@app.command()
def eval(
    experiment: str,
    split: str = 'en',
    index_type="binarizer",
):
    dataset = load_dataset('miracl/miracl', split, use_auth_token=True)

    index_path = f"experiments/miracl/{experiment}"

    index_type_enum = get_index_type(index_type)

    # lifestyle full centroids == 65536
    #lifestyle-40k-benchmark centroids == 32768
    if index_type != 'bge':
        index, collection = open_collection(index_path, index_type_enum)
    else:
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        index_type_enum = get_index_type('binarizer') # use colbert
        index, collection = open_collection(index_path, index_type_enum)

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
            opts = ldb.SearchOptions()
            opts.n_probe = 32
            opts.num_second_pass = 2500
            opts.k_top_centroids=2
            if index_type != 'bge':
                results = collection.search(0, question, 100, opts)
            else:
                import string
                query = question.translate(str.maketrans('', '', string.punctuation))
                embeds = model.encode(query, max_length=1028, return_colbert_vecs=True)['colbert_vecs']
                results = index.search(0, embeds, 100, opts)
            # if use_rerank:
            #     print("reranking...")
            #     texts = [doc.metadata['text'] for doc in results]
            #     pairs = [(question, text) for text in texts]
            #
            #     with torch.no_grad():
            #         inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            #         scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            #
            #         tups = list(zip(results, scores))
            #         results = sorted(tups, key=lambda x: x[1], reverse=True)
            #         results = [x[0] for x in results]
            #         print("done reranking...")

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
            mrr3_sum += 1/float(row[3]) if float(row[3]) <=3 else 0
            mrr5_sum += 1/float(row[3]) if float(row[3]) <=5 else 0
            DCG10_sum += float(row[4])
            iDCG10_sum += float(row[5])
            count += 1

        MRR3 = mrr3_sum / count
        MRR5 = mrr5_sum / count
        NDCG10 = DCG10_sum / iDCG10_sum

        print(f"MRR@3: {MRR3}")
        print(f"MRR@5: {MRR5}")
        print(f"NDCG@10: {NDCG10}")



@app.command()
def run(
    experiment: str,
    split: str = 'en',
    k: int = 5,
    nbits: int=1, 
    index_type="binarizer",
):
    print("Loading dataset...")
    dataset = load_dataset('miracl/miracl', split, use_auth_token=True)
    print("Dataset loaded.")

    index_path = f"experiments/miracl/{experiment}"
    # assert not os.path.exists(index_path)
    if index_path and os.path.exists(index_path):
        # delete directory if exists
        shutil.rmtree(index_path)

    if index_type != 'bge':
        index_type_enum = get_index_type(index_type)

        # lifestyle full centroids == 65536
        #lifestyle-40k-benchmark centroids == 32768
        index, collection = create_collection(index_path, index_type_enum, 128, 2)
    else:
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        index_type_enum = get_index_type('binarizer') # use colbert
        index, collection = create_collection(index_path, index_type_enum, 128, 2, num_centroids=942)


    id=0
    passages = []
    for data in tqdm(dataset['dev']):
        # query_id = data['query_id']

        passages.extend(data['positive_passages'])
        passages.extend(data['negative_passages'])

    training_data = random.sample(passages, 5000)
    training_data = [x['text'] for x in training_data]

    if index_type != 'bge':
        collection.train(training_data)
    else:

        if os.path.exists("miracl-bge-embeddings.npz"):
            print("Loading embeddings...")
            training_embeds = np.load("miracl-bge-embeddings.npz")['arr_0']
        else:
            training_embeds = None

            for sent in training_data:
                embeds = model.encode(sent, max_length=1028, return_colbert_vecs=True)['colbert_vecs']
                if training_embeds is None:
                    training_embeds = embeds
                else:
                    training_embeds = np.append(training_embeds, embeds, axis=0)

            print(np.sqrt(len(training_embeds)))
            np.savez("miracl-bge-embeddings", training_embeds)
        index.train(training_embeds)

    start = time.perf_counter()
    for passage in passages:
        if index_type != 'bge':
            collection.add(0, id, passage['text'], {
                'text': passage['text'],
                'docid': passage['docid'],
                'title': passage['title'],
            })
        else:
            embeds = model.encode(passage['text'], max_length=1028, return_colbert_vecs=True)['colbert_vecs']
            index.add(0, [{'embeddings': embeds, 'id': id, 'metadata': {'text': passage['text'], 'docid': passage['docid'], 'title': passage['title']}}])
        id += 1

    duration = time.perf_counter() - start
    print(f"Indexing complete. duration: {duration:.2f}s")

if __name__ == "__main__":
    app()