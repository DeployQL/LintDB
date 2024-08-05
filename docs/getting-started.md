# Getting Started

## Installation

To install the package, run the following command:

```bash
conda install -c deployql lintdb
```

## Usage


### Load data
Let's use LoTTE data to create a new database.

```python
from datasets import load_dataset
from collections import namedtuple

LoTTeDataset = namedtuple('LoTTeDataset', ['collection', 'queries', 'qids', 'dids'])

# get the LoTTE dataset and queries
collection_dataset = load_dataset("colbertv2/lotte_passages", 'lifestyle')
collection = [x['text'] for x in collection_dataset[split + '_collection']]
dids = [x['doc_id'] for x in collection_dataset[split + '_collection']]

queries_dataset = load_dataset("colbertv2/lotte", dataset)
queries = [x['query'] for x in queries_dataset['search_' + split]]
qids = [x['qid'] for x in queries_dataset['search_' + split]]

data = LoTTeDataset(collection, queries, qids, dids)
```

### Load a ColBERT model
We can reuse the ColBERT model from the Hugging Face model hub.

```python
from colbert.modeling.checkpoint import Checkpoint
from colbert import Searcher
config = ColBERTConfig.load_from_checkpoint("colbert-ir/colbertv2.0")
checkpoint = Checkpoint("colbert-ir/colbertv2.0", config)
```

### Create a database

LintDB requires a schema to be defined for a database.

We can create a simple ColBERT schema as follows:

```python
from lintdb.core import (
Schema,
ColbertField,
QuantizerType,
Configuration,
IndexIVF
)

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
```

Let's look at the schema we just created:

```python
ColbertField('colbert',  # field name
     DataType.TENSOR, # data type
     {
        'dimensions': 128,  # number of dimensions
        'quantization': QuantizerType.BINARIZER, # the type of quantizer to use.
        "num_centroids": 32768, # the number of centroids to use in training.
        "num_iterations": 10, # the number of iterations to use in training.
    }
)
```

ColBERT stores token embeddings of 128 dimensions. Our Quantizer `BINARIZER` is
directly translated out of the original ColBERT implementation.

The number of centroids as defined in ColBERT should be the square root of the total number of embeddings.

### Training

Before we can index data, we need to train the database to learn what
clusters to assign the embeddings to.

```python
from lintdb.core import (
Document,
TensorFieldValue
)
training_docs = []
training_data = random.sample(data.collection, min(20000, len(d.collection)))
for b in tqdm(batch(training_data, n=1000)):
    embeddings = checkpoint.docFromText(b)
    for emb in embeddings:
        emb = np.squeeze(emb.cpu().numpy().astype('float32'))
        doc = Document(0, [TensorFieldValue("colbert", emb)])
        training_docs.append(doc)

    index.train(training_docs)
```

Let's take a closer look at the Document we pass to LintDB:
```python
Document(0, # the tenant id. must be an int.
         [ # a list of field values.
             TensorFieldValue( # specifies that we're passing in tensors.
                 "colbert",  # the field name
                 emb # the embedding. must be a 2D numpy array (n, 128) of float32.
             )
         ]
         )
```
Each document must match the schema for the database.


### Indexing

Now that we have trained the database, we can index the data.

```python
 for b in tqdm(batch(list(zip(d.dids, d.collection)),n=1)):
        ids = [i for i,_ in b]
        docs = [d for _, d in b]

        embedding = checkpoint.docFromText(docs)

        e = np.squeeze(embedding.cpu().numpy().astype('float32'))

        for i, ee in zip(ids, e):
            doc = Document(i, [TensorFieldValue("colbert", e)])
            index.add(0, [doc])
```

### Searching

Now that we have indexed the data, we can search the database.

```python
from lintdb.core import (
Query,
VectorQueryNode
)
for id, query in zip(data.qids, data.queries):
    embedding = checkpoint.queryFromText(query)
    e = np.squeeze(embedding.cpu().numpy().astype('float32'))

    query = Query(
        VectorQueryNode(
            TensorFieldValue('colbert', e)
        )
    )
    results = index.search(0, query_doc, 10)
    print(results)
```