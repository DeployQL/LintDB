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
        with tempfile.TemporaryDirectory(prefix="lintdb_test-one") as dir_one:
            # create an index with 32 centroids, 128 dims, 2 bit compression, and 4 iterations during training. True == use compression.
            index_one = lintdb.IndexIVF(dir_one, 32, 128, 2, 4, 16, lintdb.IndexEncoding_NONE)

            data = np.random.rand(1500, 128).astype('float32')
            index_one.train(data)

            query = np.random.rand(30, 128).astype('float32')
            index_one.add(0,[lintdb.RawPassage(query, 1)])

            result = index_one.search(0, query, 10, 10)
            assert(len(result) == 1)

            ids = [1]
            index_one.remove(0, ids)


    def test_index_load(self):
        with tempfile.TemporaryDirectory(prefix="lintdb_test")as dir:
            index = lintdb.IndexIVF(dir, 32, 128, 2, 4, 16)

            data = np.random.rand(1500, 128).astype('float32')
            index.train(data)

            del index

            index = lintdb.IndexIVF(dir)
            search = np.random.rand(10, 128).astype('float32')
            opt = lintdb.SearchOptions()
            opt.centroid_score_threshold = 0.0
            index.search(0, search, 10, 10, opt)

    def test_index_merge(self):
        with tempfile.TemporaryDirectory(prefix="lintdb_test-one") as dir_one:
            with tempfile.TemporaryDirectory(prefix="lintdb_test-two") as dir_two:
                index_one = lintdb.IndexIVF(dir_one, 32, 128, 2, 4, 16, lintdb.IndexEncoding_BINARIZER)

                data = np.random.rand(1500, 128).astype('float32')
                index_one.train(data)

                index_two = lintdb.IndexIVF(index_one, dir_two)

                query = np.random.rand(30, 128).astype('float32')
                index_one.add(0,[lintdb.RawPassage(query, 1)])

                result = index_one.search(0, query, 10, 10)
                assert(len(result) == 1)

                index_two.add(0, [lintdb.RawPassage(query, 2)])

                # search index one againt to test we didn't add a new document to it.
                result = index_one.search(0, query, 10, 10)
                assert(len(result) == 1)
                result = index_two.search(0, query, 10, 10)
                assert(len(result) == 1)

                del index_two
                index_one.merge(dir_two)

                result = index_one.search(0, query, 10, 10)
                assert(len(result) == 2)


    def test_index_multitenancy(self):
        with tempfile.TemporaryDirectory(prefix="lintdb_test-multitenancy") as dir_one:
            # create an index with 32 centroids, 128 dims, 2 bit compression, and 4 iterations during training. True == use compression.
            index_one = lintdb.IndexIVF(dir_one, 32, 128, 2, 4, 16, lintdb.IndexEncoding_BINARIZER)

            data = np.random.rand(1500, 128).astype('float32')
            index_one.train(data)

            query = np.random.rand(30, 128).astype('float32')
            index_one.add(0,[lintdb.RawPassage(query, 1)])
            index_one.add(2,[lintdb.RawPassage(query, 3)])

            result = index_one.search(0, query, 10, 10)
            assert(len(result) == 1)

            assert(result[0].id == 1)

            # search the other tenant
            result = index_one.search(2, query, 10, 10)
            assert(len(result) == 1)

            assert(result[0].id == 3)


if __name__ == '__main__':
    unittest.main()