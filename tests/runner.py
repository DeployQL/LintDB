import lintdb
import numpy as np
import time
import os
from tqdm import tqdm

DB_PATH = "experiments/test_bench4"

def main():
    if not os.path.exists(DB_PATH):
        index = lintdb.IndexIVF(DB_PATH, 128, 128, 1, 4)

        training_data = np.random.rand(5000, 128).astype('float32')
        index.train(training_data)

        add_latencies = []

        for i in tqdm(range(100000)):
            doc = np.random.rand(120, 128).astype('float32')
            start = time.perf_counter()
            dd = lintdb.RawPassage(doc, i+1)
            index.add(0, [dd])
            add_latencies.append(start-time.perf_counter())

        index.flush()

        print(f"p95 add latency: {np.percentile(add_latencies, 95):.2f}")
    else:
        index = lintdb.IndexIVF(DB_PATH)

    query = np.random.rand(32, 128).astype("float32")

    # for i in range(65000):
    #     pids = index.lookup_pids(0, i)
    #     print(pids)

    start = time.perf_counter()
    options = lintdb.SearchOptions()
    options.centroid_score_threshold = 0.
    options.k_top_centroids=2
    # options.expected_id= 1

    result = index.search(0, query, 32, 100, options)
    end = time.perf_counter()
    print(f"results: {len(result)}")
    

    # query = np.random.rand(100, 128).astype('float32')
    # clusters = np.random.rand(65000, 128).astype('float32')

    # start = time.perf_counter()
    # result = np.dot(clusters, np.transpose(clusters)) # 57ms
    # end = time.perf_counter()

    print(f"duration: {end-start}")
    # index.search(0, dat, 32, 100)


main()