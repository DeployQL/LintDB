import lintdb as ldb
from lintdb.schema import Schema, Colbert, DataType, Quantizer_Binarizer
from lintdb.document import Document
from lintdb.lintdb import Configuration, IndexIVF
from colbert.infra import Run, RunConfig, ColBERTConfig
from tqdm import tqdm
import shutil
import numpy as np
import typer

app = typer.Typer()

@app.command()
def run():
    print("here")
    index_path = "local_db.index"
    checkpoint = "colbert-ir/colbertv2.0"


    from colbert.modeling.checkpoint import Checkpoint
    from colbert import Searcher
    config = ColBERTConfig.load_from_checkpoint(checkpoint)
    checkpoint = Checkpoint(checkpoint, config)

    print("here")
    schema = Schema(
        [
            Colbert('colbert', DataType.TENSOR, {
                'dimensions': 128,
                'quantization': Quantizer_Binarizer
            })
        ]
    )
    print("here")
    config = Configuration()
    print("here")
    index = IndexIVF(index_path, schema, config)
    print("here")

if __name__ == "__main__":
    app()
