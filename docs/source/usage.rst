Usage
=====

Create your first database
===

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
    results = index.search(0, query, 5) # search the database for the 5 most similar documents to the query

    # delete documents from the database
    index.remove(
        0, # tenant id
        [0, 1, 2] # list of document ids to delete
    )


Load an existing database
===

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
