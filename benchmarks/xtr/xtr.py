import collections
import numpy as np
import nltk
import os
import pytrec_eval
import re
import time
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
import transformers

from dataclasses import dataclass
from enum import Enum
from nltk.tokenize import sent_tokenize
from typing import List
from tqdm import tqdm
def profile(func, debug=DEBUG):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        if debug:
            print(f"{func.__name__} took {time.time() - started_at:.3f} seconds.")
        return result
    return wrap


class XTR(object):
    def __init__(self, encoder, model_type, index_type='faiss'):
        self.encoder = encoder
        self.index_type = index_type  # must be 'faiss' or 'scann'. Otherwise uses bruteforce.

        # Set the tokenizer based on the model type.
        if model_type not in MULTILINGUAL_MODELS:
            with tf.io.gfile.GFile("gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model", "rb") as f:
                self.tokenizer = text.SentencepieceTokenizer(model=f.read(), add_eos=True)
        else:
            with tf.io.gfile.GFile("gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model", "rb") as f:
                self.tokenizer = text.SentencepieceTokenizer(model=f.read(), add_eos=True)

    def tokenize(self, text):
        return [self.tokenizer.id_to_string(id_).numpy().decode('utf-8') for id_ in self.tokenizer.tokenize(text)]

    @profile
    def get_token_embeddings(self, texts):
        batch_embeds = self.encoder(tf.constant([t.lower() for t in texts]))
        batch_lengths = np.sum(batch_embeds["mask"].numpy(), axis=1)
        return batch_embeds["encodings"].cpu().numpy(), batch_lengths

    @profile
    def get_flatten_embeddings(self, batch_text, return_last_offset=False):
        batch_embeddings, batch_lengths = self.get_token_embeddings(batch_text)
        flatten_embeddings = None
        num_tokens = 0
        offsets = [0]
        for batch_id, (embeddings, length) in enumerate(zip(batch_embeddings, batch_lengths)):
            if flatten_embeddings is not None:
                flatten_embeddings = np.append(flatten_embeddings, embeddings[:int(length)], axis=0)
            else:
                flatten_embeddings = embeddings[:int(length)]
            num_tokens += int(length)
            offsets.append(num_tokens)
        assert num_tokens == flatten_embeddings.shape[0]
        if not return_last_offset:
            offsets = offsets[:-1]
        return flatten_embeddings, offsets

    @profile
    def build_index(self, documents, batch_size=32):
        all_token_embeds = np.zeros((len(documents)*MAX_SEQ_LEN, TOKEN_EMBED_DIM), dtype=np.float32)
        all_doc_offsets = []
        num_tokens = 0
        for batch_idx in tqdm(range(0, len(documents), batch_size)):
            batch_docs = documents[batch_idx:batch_idx+batch_size]
            batch_embeds, batch_offsets = self.get_flatten_embeddings(batch_docs)
            all_doc_offsets += [num_tokens + offset for offset in batch_offsets]
            num_tokens += len(batch_embeds)
            all_token_embeds[num_tokens-len(batch_embeds):num_tokens] = batch_embeds

        # Use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher.
        if self.index_type == 'scann':
            self.searcher = scann.scann_ops_pybind.builder(all_token_embeds[:num_tokens], 10, "dot_product").tree(
                num_leaves=min(2000, num_tokens), num_leaves_to_search=100, training_sample_size=min(250000, num_tokens)).score_ah(
                1, anisotropic_quantization_threshold=0.1).build()
        elif self.index_type == 'faiss':
            ds = 128
            num_clusters = 50
            code_size = 64
            quantizer = faiss.IndexFlatIP(ds)
            opq_matrix = faiss.OPQMatrix(ds, code_size)
            opq_matrix.niter = 10
            sub_index = faiss.IndexIVFPQ(quantizer, ds, num_clusters, code_size, 4, faiss.METRIC_INNER_PRODUCT)
            index = faiss.IndexPreTransform(opq_matrix, sub_index)
            index.train(all_token_embeds[:num_tokens])
            index.add(all_token_embeds[:num_tokens])
            class FaissSearcher(object):
                def __init__(self, index):
                    self.index = index
                def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
                    scores, top_ids = self.index.search(query_embeds, final_num_neighbors)
                    return top_ids, scores
            self.searcher = FaissSearcher(index)
        # Used only for small-scale, exact inference.
        else:
            class BruteForceSearcher(object):
                def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
                    scores = query_embeds.dot(all_token_embeds[:num_tokens].T) # Q x D
                    top_ids = scores.argsort(axis=1)[:, ::-1][:,:final_num_neighbors] # Q x top_k
                    return top_ids, [q_score[q_top_ids] for q_score, q_top_ids in zip(scores, top_ids)] # (Q x top_k, Q x top_k)
            self.searcher = BruteForceSearcher()

        self.doc_offsets = all_doc_offsets
        self.doc_offsets.append(num_tokens)  # Add final number of tokens.
        self.tid2did = {
            self.doc_offsets[did] + tid: did
            for did in range(len(self.doc_offsets)-1)
            for tid in range(self.doc_offsets[did+1] - self.doc_offsets[did])
        }
        self.tid2did[-1] = 0
        self.docs = documents
        print("Index Ready!", self.searcher)

    @profile
    def batch_search_tokens(self, batch_query, token_top_k=100, leaves_to_search=100, pre_reorder_num_neighbors=100):
        all_query_encodings, query_offsets = self.get_flatten_embeddings(batch_query, return_last_offset=True)
        all_neighbors, all_scores = self.searcher.search_batched(
            all_query_encodings, final_num_neighbors=token_top_k, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=pre_reorder_num_neighbors
        )
        return [
            (
                [f'q_{i}' for i in range(query_offsets[oid], query_offsets[oid+1])],  # query_id
                all_neighbors[query_offsets[oid]:query_offsets[oid+1]],  # neighbors
                all_scores[query_offsets[oid]:query_offsets[oid+1]],  # scores
            )
            for oid in range(len(query_offsets)-1)
        ]

    @profile
    def estimate_missing_similarity(self, batch_result):
        batch_qtoken_to_ems = [dict() for _ in range(len(batch_result))]
        for b_idx, (query_tokens, _, distances) in enumerate(batch_result):
            for token_idx, qtoken in enumerate(query_tokens):
                idx_t = (token_idx, qtoken)
                # Use similarity of the last token as imputed similarity.
                batch_qtoken_to_ems[b_idx][idx_t] = distances[token_idx][-1]
        return batch_qtoken_to_ems

    def aggregate_scores(self, batch_result, batch_ems, document_top_k):
        """Aggregates token-level retrieval scores into query-document scores."""

        @profile
        def get_did2scores(query_tokens, all_neighbors, all_scores):
            did2scores = {}
            # |Q| x k'
            for qtoken_idx, (qtoken, neighbors, scores) in enumerate(zip(query_tokens, all_neighbors, all_scores)):
                for _, (doc_token_id, score) in enumerate(zip(neighbors, scores)):
                    if np.isnan(score):
                        continue
                    docid = self.tid2did[doc_token_id]
                    if docid not in did2scores:
                        did2scores[docid] = {}
                    qtoken_with_idx = (qtoken_idx, qtoken)
                    if qtoken_with_idx not in did2scores[docid]:
                        # Only keep the top score for sum-of-max.
                        did2scores[docid][qtoken_with_idx] = score

            return did2scores
        batch_did2scores = [get_did2scores(qtokens, neighbors, scores) for qtokens, neighbors, scores in batch_result]

        @profile
        def add_ems(did2scores, query_tokens, ems):
            # |Q| x |Q|k' (assuming most docid is unique)
            for qtoken_idx, qtoken in enumerate(query_tokens):
                qtoken_with_idx = (qtoken_idx, qtoken)
                for docid, scores in did2scores.items():
                    if qtoken_with_idx not in scores:
                        scores[qtoken_with_idx] = ems[qtoken_with_idx]
        for did2scores, result, ems in zip(batch_did2scores, batch_result, batch_ems):
            add_ems(did2scores, result[0], ems)

        @profile
        def get_final_score(did2scores, query_tokens):
            final_qd_score = {}
            # |Q|k' x |Q|
            for docid, scores in did2scores.items():
                assert len(scores) == len(query_tokens)
                final_qd_score[docid] = sum(scores.values()) / len(scores)
            return final_qd_score

        batch_scores = [get_final_score(did2scores, result[0]) for did2scores, result in zip(batch_did2scores, batch_result)]

        batch_ranking = [
            sorted([(docid, score) for docid, score in final_qd_score.items()], key=lambda x: x[1], reverse=True)[:document_top_k]
            for final_qd_score in batch_scores
        ]
        return batch_ranking

    def get_document_text(self, batch_ranking):
        batch_retrieved_docs = []
        for ranking in batch_ranking:
            retrieved_docs = []
            for did, score in ranking:
                retrieved_docs.append((did, score, self.docs[did]))
            batch_retrieved_docs.append(retrieved_docs)
        return batch_retrieved_docs

    def retrieve_docs(
            self,
            batch_query: List[str],
            token_top_k: int = 100,
            leaves_to_search: int = 100,
            pre_reorder_num_neighbors: int = 100,
            document_top_k: int = 100,
            return_text: bool = True,
    ):
        """Runs XTR retrieval for a query."""
        batch_result = self.batch_search_tokens(batch_query, token_top_k=token_top_k, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=pre_reorder_num_neighbors)
        batch_mae = self.estimate_missing_similarity(batch_result)
        batch_ranking = self.aggregate_scores(batch_result, batch_mae, document_top_k)
        if return_text:
            return self.get_document_text(batch_ranking), batch_result
        else:
            return batch_ranking, batch_result