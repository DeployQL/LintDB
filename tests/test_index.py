from pylintdb import pylintdb
import numpy as np

def test_index_init():
    index = pylintdb.IndexIVF("/tmp/py_index", 32, 128, 1)

    passages = []
    for i in range(10):
        data = np.random.rand(100, 128).astype('float32')
        obj = pylintdb.RawPassage(
            data,
            i,
            "doc id",
        )
        passages.append(obj)

    data = np.random.rand(1500, 128).astype('float32')
    index.train(data)

    # for i in range(2):
    #     pass
    ids = list(range(5))
    index.remove(ids)

    dat = np.random.rand(100, 128).astype('float32')
    obj = pylintdb.RawPassage(
            dat,
            1,
            "doc id",
        )
    # I can append to our own vector type, but can't be added for some reason.
    td = pylintdb.RawPassageVector()
    for i in range(5):
        td.append(pylintdb.RawPassage(dat, 1, "doc id"))

    index.add_single(obj)

    index.add(passages)