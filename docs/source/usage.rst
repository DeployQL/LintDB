Usage
=====

--------------------------
Create your first database
--------------------------

We can demonstrate LintDB with fake data.  This will show how to:
1. Create a new database.
2. Train the database.
3. Add documents to the database.
4. Search the database.
5. Remove documents from the database.

.. code-block:: python

    import lintdb
    import numpy as np
    import tempfile

    db_dir = "path/to/your/db"
    index = lintdb.IndexIVF(
        db_dir, # path to your database to create
        5, # number of clusters to use in training
        128, # dimensions of the vectors
        1,  # number of bits to use in vector compression
        10 # number of iterations used in training
    )

    data = np.random.normal(5, 5, size=(1500, 128)).astype('float32')
    # train() accepts a numpy array of vectors. Each row will be used in training.
    index.train(data)

    passages = []
    for i in range(10):
        data = np.random.normal(i, 2, size=(100, 128)).astype('float32')
        obj = lintdb.RawPassage(
            data, # vector data for the document
            i # unique id for the document. must be an integer.
        )
        passages.append(obj)

    index.add(
        0,  # LintDB supports tenants. the tenant id is an integer.
        passages # array of documents.
    )


    # search the database
    query = np.random.normal(5, 5, size=(1, 128)).astype('float32')

    # results is a list of tuples. Each tuple contains the document id and the distance from the query.
    results = index.search(
        0, 
        query, 
        5, # nprobe: number of centroids to search across.
        10 # k: number of documents to return
    ) 

    # delete documents from the database
    index.remove(
        0, # tenant id
        [0, 1, 2] # list of document ids to delete
    )


-------------------------
Load an existing database
-------------------------

We can also load an existing database given a path.

.. code-block:: python

    import lintdb
    import numpy as np
    import tempfile

    db_dir = "path/to/your/db"
    index = lintdb.IndexIVF(db_dir)

    # search the database
    query = np.random.normal(5, 5, size=(1, 128)).astype('float32')

    # results is a list of tuples. Each tuple contains the document id and the distance from the query.
    results = index.search(0, query, 5) # search the database for the 5 most similar documents to the query
    # results object is a list of lintdb.SearchResult;
    # results[0].id
    # results.[0].score


-------------------------
Train (or reuse) an index
-------------------------

Train the index
^^^^^^^^^^^^^^^^

LintDB's training procedure follows ColBERT's approach. We use `faiss <https://github.com/facebookresearch/faiss>`
to train k centroids from training data.

If compression is enabled, then LintDB also learns buckets in order to compress residuals into bit vectors.

.. code-block:: python

    training_data = []
    sample = random.sample(dataset.collection, 25000)
    for doc in tqdm(sample, desc="embedding training data"):
        embeddings = checkpoint.docFromText([doc])
        training_data.append(np.squeeze(embeddings.numpy().astype('float32')))

    dd = np.concatenate(training_data).astype('float32')

    index.train(dd)


Reuse the trained weights from an existing index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can reuse weights from a previously trained ColBERT model with the following snippet.

.. code-block:: python

    index = lintdb.IndexIVF(index_path, 65536, 128, nbits, 10, use_compression)

    with Run().context(RunConfig(nranks=1, experiment='colbert-lifestyle-full')):
        checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
        config = ColBERTConfig.from_existing(checkpoint_config, None)
        searcher = Searcher(index='colbert-lifestyle-full', config=config, collection=d.collection)
        
        centroids = searcher.ranker.codec.centroids

        index.set_centroids(centroids)
        index.set_weights(
            searcher.ranker.codec.bucket_weights.tolist(), 
            searcher.ranker.codec.bucket_cutoffs.tolist(), 
            searcher.ranker.codec.avg_residual
        )
        index.save()