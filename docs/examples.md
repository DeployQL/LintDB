# Examples

## ColPali: Processing PDFs 

ColPali is a recent advancement in efficient document retrieval. Instead of using a traditional document
processing pipeline to extract data, ColPali takes embeddings of the image of the PDFs. See: [ColPali](https://github.com/illuin-tech/colpali)

### Install Dependencies

```python
conda install -c deployql lintdb

# make sure to update the url depending on your cuda version.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers tqdm  git+https://github.com/illuin-tech/colpali
```

### Login to Huggingface

ColPali is built on top of PaliGemma, which requires user agreement in the HF model hub. 

To login to the Huggingface model hub, see here: [Huggingface Gated Models Documentation](https://huggingface.co/docs/hub/en/models-gated#download-files)

### Define our imports

```python
from lintdb.core import (
    Schema,
    ColbertField,
    QuantizerType,
    Binarizer,
    Configuration,
    SearchOptions,
    DataType,
    FieldValue,
    TensorFieldValue,
    Document,
    IndexIVF,
    VectorQueryNode,
    Query
)
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image
import numpy as np

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import load_from_dataset
```

We import the necessary libraries to run ColPali, including the `lintdb` library for indexing and searching, and the `colpali_engine` library for processing.

### Load the model

```python
model_name = "vidore/colpali"
model = ColPali.from_pretrained("google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cuda").eval()
model.load_adapter(model_name)
processor = AutoProcessor.from_pretrained(model_name)
```

### Load sample data

```python
images = load_from_dataset("vidore/docvqa_test_subsampled")
queries = ["From which university does James V. Fiorca come ?", "Who is the japanese prime minister?"]

dataloader = DataLoader(
    images,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: process_images(processor, x),
)
ds = []
for batch_doc in tqdm(dataloader):
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc)
    ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
```

### Index the data

```python
# train the index
num_embeddings = sum([x.type(torch.FloatTensor).numpy().shape[0] for x in ds])

schema = Schema(
    [
        ColbertField('colbert', DataType.TENSOR, {
            'dimensions': 128,
            'quantization': QuantizerType.BINARIZER,
            "num_centroids": 717,
            "num_iterations": 10,
        })
    ]
)
config = Configuration()
index = IndexIVF('vidore-sample.db', schema, config)


training_docs = []
for emb in ds:
    emb = np.squeeze(emb.type(torch.FloatTensor).numpy().astype('float32'))
    doc = Document(0, [TensorFieldValue("colbert", emb)])
    training_docs.append(doc)
index.train(training_docs)

# index the documents.
for i, emb in enumerate(ds):
    emb = np.squeeze(emb.type(torch.FloatTensor).numpy().astype('float32'))
    doc = Document(i, [TensorFieldValue("colbert", emb)])
    index.add(0, [doc])
```

### Query the data

```python
for query in qs:
  root = VectorQueryNode(TensorFieldValue('colbert', query.type(torch.FloatTensor).numpy().astype('float32')))
  query = Query(root)
  results = index.search(0, query, 10, {
      'n_probe': 32,
      'colbert_field': 'colbert',
      'k_top_centroids': 2,
  })
  for i in range(5):
    print(results[i].id)
  print("----")
```