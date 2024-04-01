import unittest
import lintdb
import numpy as np
import tempfile

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class TestIndex(unittest.TestCase):
    def test_index_init(self):
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

        index.add(0, passages)

        index.search(0, dat, 10, 100)


        ids = list(range(5))
        index.remove(0, ids)


    def test_index_load(self):
        dir = tempfile.TemporaryDirectory(prefix="lintdb_test")
        index = lintdb.IndexIVF(dir.name, 32, 128, 2, 4, False)

        data = np.random.rand(1500, 128).astype('float32')
        index.train(data)

        del index

        index = lintdb.IndexIVF(dir.name)
        search = np.random.rand(10, 128).astype('float32')
        index.search(0, search, 10, 10)

    def test_index_merge(self):
        dir_one = tempfile.TemporaryDirectory(prefix="lintdb_test")
        index_one = lintdb.IndexIVF(dir_one.name, 32, 128, 2, 4, True)

        data = np.random.rand(1500, 128).astype('float32')
        index_one.train(data)

        dir_two = tempfile.TemporaryDirectory(prefix="lintdb_test")
        index_two = lintdb.IndexIVF(index_one, dir_two.name)

        query = np.random.rand(30, 128).astype('float32')
        index_one.add(0, lintdb.RawPassage(query, 1))

        result = index_one.search(0, query, 10, 10)
        assert(len(result) == 1)

        index_two.add(0, lintdb.RawPassage(query, 2))

        # search index one againt to test we didn't add a new document to it.
        result = index_one.search(0, query, 10, 10)
        assert(len(result) == 1)

        index_one.merge(index_two)

        # test that after merging, we get two results.
        result = index_one.search(0, query, 10, 10)
        assert(len(result) == 2)



        





if __name__ == '__main__':
    unittest.main()