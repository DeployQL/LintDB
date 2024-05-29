import unittest
import lintdb
import numpy as np
import tempfile
import os
import urllib.request

def get_file_if_not_exists(url, file):
    if not os.path.exists(file):
        urllib.request.urlretrieve(url, file)

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
            # create an index with 32 centroids, 128 dims, 2 bit compression, and 4 iterations during training. 
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

    def test_collection(self):
        # these exist because in conda, we don't download the files using cmake.
        get_file_if_not_exists("https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx", "model.onnx")
        get_file_if_not_exists("https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/tokenizer.json", "colbert_tokenizer.json")
        
        with tempfile.TemporaryDirectory(prefix="lintdb_test-collection") as dir_one:
            # create an index with 32 centroids, 128 dims, 2 bit compression, and 4 iterations during training.
            index_one = lintdb.IndexIVF(dir_one, 32, 128, 2, 4, 16, lintdb.IndexEncoding_BINARIZER)

            collection_options = lintdb.CollectionOptions()
            collection_options.model_file = "model.onnx"
            collection_options.tokenizer_file = "colbert_tokenizer.json"
            collection = lintdb.Collection(index_one, collection_options)

            collection.train(['hello world!'] * 1500)

            collection.add(0, 1, "hello world!", {"title": "metadata"})

            opts = lintdb.SearchOptions()
            opts.n_probe = 250
            results = collection.search(0, "hello world!", 10, opts)


            assert(len(results) == 1)
            assert(results[0].id == 1)
            assert(results[0].metadata['title'] == 'metadata')

    def test_collection_batch(self):
        get_file_if_not_exists("https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx", "model.onnx")
        get_file_if_not_exists("https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/tokenizer.json", "colbert_tokenizer.json")
        
        with tempfile.TemporaryDirectory(prefix="lintdb_test-collection") as dir_one:
            # create an index with 32 centroids, 128 dims, 2 bit compression, and 4 iterations during training.
            index_one = lintdb.IndexIVF(dir_one, 32, 128, 2, 4, 16, lintdb.IndexEncoding_BINARIZER)

            collection_options = lintdb.CollectionOptions()
            collection_options.model_file = "model.onnx"
            collection_options.tokenizer_file = "colbert_tokenizer.json"
            collection = lintdb.Collection(index_one, collection_options)

            collection.train(['hello world!'] * 1500)
            
            opts = lintdb.SearchOptions()
            opts.n_probe = 250

            collection.add_batch(0, [
                {"id": 1, "text": "hello world!", "metadata": {"title": "metadata"}},
                {"id": 2, "text": "hello world!", "metadata": {"title": "metadata"}}
            ])

            results = collection.search(0, "hello world!", 10)


            assert(len(results) == 2)



if __name__ == '__main__':
    unittest.main()