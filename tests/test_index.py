import lintdb
import numpy as np
import tempfile

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def test_index_init():
    np.random.seed(seed=12345)
    db_dir = tempfile.TemporaryDirectory(prefix="lintdb_test")
    config = lintdb.Configuration()
    index = lintdb.IndexIVF(db_dir.name, 5, 128, 1, 10)

    passages = []
    for i in range(10):
        data = np.random.normal(i, 2, size=(100, 128)).astype('float32')
        obj = lintdb.RawPassage(
            data,
            i
        )
        passages.append(obj)

    data = np.random.normal(5, 5, size=(1500, 128)).astype('float32')
    normed_data = normalized(data)
    index.train(normed_data)

    dat = np.random.normal(2, 1, size=(100, 128)).astype('float32')
    normed_dat = normalized(dat)
    obj = lintdb.RawPassage(
            normed_dat,
            1
        )

    # index.add_single(obj)

    index.add(0, passages)

    # index.search(dat, 10, 100)


    ids = list(range(5))
    index.remove(0, ids)


def test_index_load():
    dir = tempfile.TemporaryDirectory(prefix="lintdb_test")
    index = lintdb.IndexIVF(dir.name, 32, 128, 2, 4, False)

    data = np.random.rand(1500, 128).astype('float32')
    index.train(data)

    del index

    index = lintdb.IndexIVF(dir.name)
    search = np.random.rand(10, 128).astype('float32')
    index.search(0, search, 10, 10)

test_index_init()