import numpy as np
import os
import shutil
import tempfile
import unittest

from lintdb.core import (
    Schema,
    ColbertField,
    QuantizerType,
    Binarizer,
    Configuration,
    FaissCoarseQuantizer,
    DataType,
    TensorFieldValue,
    QuantizedTensorFieldValue,
    IntFieldValue,
    FloatFieldValue,
    TextFieldValue,
    DateFieldValue,
    Document,
    IndexIVF,
    Version,
    VectorQueryNode,
    Query
)


class TestIndex(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory after each test."""
        shutil.rmtree(self.test_dir)

    def test_index(self):
        schema = Schema(
            [
                ColbertField('colbert', DataType.TENSOR, {
                    'dimensions': 128,
                    'quantization': QuantizerType.BINARIZER
                })
            ]
        )
        config = Configuration()
        index = IndexIVF(self.test_dir, schema, config)

    def test_remove(self):
        schema = Schema(
            [
                ColbertField('colbert', DataType.TENSOR, {
                    'dimensions': 128,
                    'quantization': QuantizerType.BINARIZER,
                    'num_centroids': 10,
                    'num_iterations': 10
                })
            ]
        )
        config = Configuration()
        index = IndexIVF(self.test_dir, schema, config)

        docs = []
        for i in range(400):
            data = np.random.rand(30, 128).astype('float32')
            docs.append(Document(i, [TensorFieldValue('colbert', data)]))

        index.train(docs)

        # takes a list of Documents. Each document is a list of field values.
        docs = [
            Document(0, [TensorFieldValue('colbert', np.random.rand(30, 128).astype('float32'))]),
            Document(1, [TensorFieldValue('colbert', np.random.rand(30, 128).astype('float32'))])
        ]

        # add(tenant, documents)
        index.add(0, docs)

        root = VectorQueryNode(TensorFieldValue('colbert', np.random.rand(30, 128).astype('float32')))
        query = Query(root)
        results = index.search(0, query, 10, {
            'n_probe': 50,
            'colbert_field': 'colbert',
            'k_top_centroids': 100,
        })

        self.assertEqual(len(results), 2, "results should have 2 elements")

        index.remove(0, [0])

        root = VectorQueryNode(TensorFieldValue('colbert', np.random.rand(30, 128).astype('float32')))
        query = Query(root)
        results = index.search(0, query, 10, {
            'n_probe': 50,
            'colbert_field': 'colbert',
            'k_top_centroids': 100,
        })

        self.assertEqual(len(results), 1, "result should be empty")

    def test_add(self):
        schema = Schema(
            [
                ColbertField('colbert', DataType.TENSOR, {
                    'dimensions': 128,
                    'quantization': QuantizerType.BINARIZER,
                    'num_centroids': 10,
                    'num_iterations': 10
                })
            ]
        )
        config = Configuration()
        index = IndexIVF(self.test_dir, schema, config)

        docs = []
        for i in range(400):
            data = np.random.rand(30, 128).astype('float32')
            docs.append(Document(i, [TensorFieldValue('colbert', data)]))

        index.train(docs)

        # takes a list of Documents. Each document is a list of field values.
        docs = [Document(0, [TensorFieldValue('colbert', np.random.rand(30, 128).astype('float32'))])]

        # add(tenant, documents)
        index.add(0, docs)

        root = VectorQueryNode(TensorFieldValue('colbert', np.random.rand(30, 128).astype('float32')))
        query = Query(root)
        results = index.search(0, query, 10, {
            'n_probe': 50,
            'colbert_field': 'colbert',
            'k_top_centroids': 100,
        })

        self.assertEqual(len(results), 1, "results should have 1 element")

    def test_binarizer(self):
        """Test Binarizer functionality."""
        binarizer = Binarizer([0.5, 0.5], [0.5, 0.5], 0.3, 1, 128)

        # Test saving and loading
        save_path = os.path.join(self.test_dir, 'binarizer.bin')
        binarizer.save(save_path)

    def test_coarse_quantizer(self):
        """Test CoarseQuantizer functionality. We should be able to import centroids from a numpy array"""
        quantizer = FaissCoarseQuantizer(np.random.rand(10, 128).astype(np.float32))

        # Test saving and loading
        save_path = os.path.join(self.test_dir, 'quantizer.bin')
        quantizer.save(save_path)


    def test_field_value(self):
        """Test FieldValue functionality."""
        int_field = IntFieldValue('intfield', 42)
        float_field = FloatFieldValue('floatfield', 3.14)
        text_field = TextFieldValue('textfield', "example")

        tensor_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_field = TensorFieldValue('tensorfield', tensor_data)

        quantized_tensor_data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        quantized_tensor_field = QuantizedTensorFieldValue('quantizedtensorfield', quantized_tensor_data)

        from datetime import datetime
        epoch = datetime.utcfromtimestamp(0)
        now = datetime.utcnow()
        millis = int((now - epoch).total_seconds() * 1000)
        date_field = DateFieldValue('datefield', millis)

        self.assertEqual(int_field.data_type, DataType.INTEGER)
        self.assertEqual(float_field.data_type, DataType.FLOAT)
        self.assertEqual(text_field.data_type, DataType.TEXT)
        self.assertEqual(tensor_field.data_type, DataType.TENSOR)
        self.assertEqual(quantized_tensor_field.data_type, DataType.QUANTIZED_TENSOR)
        self.assertEqual(date_field.data_type, DataType.DATETIME)