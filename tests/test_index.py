from pylintdb import pylintdb
import numpy as np
import tempfile

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

    index.search(dat, 10, 100)


    ids = list(range(5))
    index.remove(ids)


def test_index_load():
    # dir = tempfile.TemporaryDirectory()
    # index = pylintdb.IndexIVF(dir.name, 32, 128, 1)

    # data = np.random.rand(1500, 128).astype('float32')
    # index.train(data)

    # del index

    s = '/tmp/py_index_bench'
    index = pylintdb.IndexIVF(s)
    search = np.random.rand(10, 128).astype('float32')
    index.search(search, 10, 10)