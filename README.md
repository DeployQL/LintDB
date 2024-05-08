![icon](icon.svg)

LintDB
=========

**LintDB** is a multi-vector database meant for Gen AI. LintDB natively supports late interaction like colBERT and PLAID.

# Key Features
- **Multi vector support**: LintDB stores multiple vectors per document id and calculates the max similarity across vectors to determine relevance. 
- **Bit-level Compression**: LintDB fully implements PLAID's bit compression, storing 128 dimension embeddings in as low as 16 bytes.  
- **Embedded**: LintDB can be embedded directly into your Python application. No need to setup a separate database.  
- **Full Support for PLAID and ColBERT**: LintDB is built around PLAID and colbert
for efficient storage and lookup of token level embeddings.

# Installation
LintDB relies on OpenBLAS for accerlated matrix multiplication. To smooth the process of installation, we only support conda.

```
conda install lintdb -c deployql -c conda-forge
```

## Usage
LintDB makes it easy to upload data, even if you have multiple tenants.

```python
index = ldb.IndexIVF(index_path)
...
# we use an IVF index, so we need to train the centroids.
index.train(training_data)
...
# add documents to the index.
doc = ldb.RawPassage(embeddings, id)
index.add(tenant_id, [doc])

results = index.search(
    tenant_id,
    embeddings, 
    32, # number of centroids to search
    100, # k to return
)
```

# Roadmap

LintDB aims to be a full retrieval platform. 

We want to extend LintDB's features to include:
- Snippet highlighting and explainability features.
- Support for more algorithms for retrieval and ranking.
    - [XTR](https://arxiv.org/pdf/2304.01982.pdf)
    - Fine tuning and pretraining, like [PreFLMR](https://arxiv.org/pdf/2402.08327.pdf)
- Increased support for document filtering.

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
