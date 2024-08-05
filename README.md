![icon](icon.svg)

LintDB
=========

**LintDB** is a multi-vector database meant for Gen AI. LintDB natively supports late interaction like ColBERT and PLAID.

# Key Features
- **Multi vector support**: LintDB stores multiple vectors per document id and calculates the max similarity across vectors to determine relevance. 
- **Bit-level Compression**: LintDB fully implements PLAID's bit compression, storing 128 dimension embeddings in as low as 16 bytes.  
- **Embedded**: LintDB can be embedded directly into your Python application. No need to setup a separate database.  
- **Full Support for PLAID and ColBERT**: LintDB is built around PLAID and ColBERT.
- **Filtering**: LintDB supports filtering on any field in the schema.

# Installation
LintDB relies on OpenBLAS for accerlated matrix multiplication. To smooth the process of installation, we only support conda.

```
conda install lintdb -c deployql -c conda-forge
```

## Usage
LintDB makes it easy to upload data, even if you have multiple tenants.

Below shows creating a database. LintDB defines a schema for a given database that can be used
to index embeddings, floats, strings, even dates. Fields can be indexed, stored, or used as a filter.
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
)
```

And querying the database. We can query any of the data fields we indexed.
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
results = index.search(0, query, 10)
print(results)
```

## Late Interaction Model Support
LintDB aims to support late interaction and more advanced retrieval models. 

- [x] ColBERTv2 with PLAID
- [x] XTR (experimental)

# Roadmap

LintDB aims to be a retrieval platform for Gen AI.
We believe that to do this, we must support flexible retrieval and scoring methods while
maintaining a high level of performance.

- Improving performance and scalability
- Improved benchmarks
- Support CITADEL for scalable late interaction
- Support learnable query adapters in the retrieval pipeline
- Enhance support for arbitrary retrieval and ranking functions
- Support learnable ranking functions

# Comparison with other Vector Databases
LintDB is one of two databases that support token level embeddings. The other being Vespa.
## Token Level Embeddings

### Vespa
Vespa is a robust, mature search engine with many features. However, the learning curve to get started and operate Vespa is high.
With embedded LintDB, there's no setup required. `conda install lintdb -c deployql` and get started.

## Embedded
### Chroma
Chroma is an embedded vector database available in Python and Javascript. LintDB currently only supports Python. 

However, unlike Chroma, LintDB offers multi-tenancy support.

# Documentation
For detailed documentation on using LintDB, refer to the [official documentation](https://deployql.github.io/LintDB/index.html)

# License
LintDB is licensed under the Apache 2.0 License. See the LICENSE file for details.

# We want to offer a managed service
We need your help! If you'd want a managed LintDB, reach out and let us know. 

Book time on the founder's calendar: https://calendar.app.google/fsymSzTVT8sip9XX6
