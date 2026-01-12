import os
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pytrec_eval
from tqdm import tqdm
import logging
import torch
import collections
import json
import random
from datetime import datetime
from pathlib import Path
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(
            model_name_or_path=model_path, 
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
                },
            trust_remote_code=True
            )
        self.model.max_seq_length = 8192
        self.pool = self.model.start_multi_process_pool()

    def _load_data(self, queries_path: str, corpus_path: str, qrels_path: str) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
        logger.info(f"Loading data from {queries_path}, {corpus_path}, {qrels_path}")
        qid2eval_res = {}
        queries = pl.read_parquet(queries_path).rows(named=True)
        for item in queries:
            qid2eval_res[item["_id"]] = {"pos_char_span": item["pos_char_span"], "pos_char_length": item["pos_char_length"], "pos_token_length": item["pos_token_length"]}

        corpus = pl.read_parquet(corpus_path).rows(named=True)
        qrels = pl.read_parquet(qrels_path)

        logger.info(f"Found {len(queries)} queries, {len(corpus)} corpus documents, {len(qrels)} qrels")

        valid_query_ids = set(row['_id'] for row in queries)
        qrels = qrels.filter(pl.col('query-id').is_in(valid_query_ids)).rows(named=True)

        logger.info(f"Loaded {len(valid_query_ids)} valid query ids, {len(qrels)} qrels")

        return queries, corpus, qrels, qid2eval_res

    def _encode_texts(self, texts: List[str], batch_size: int = 32, is_query=False) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Texts to encode, list of strings
            batch_size: Batch size, default 32

        Returns:
            Embeddings, shape (n_texts, embedding_dim)
        """
        logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
        task_instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

        if is_query:
            texts = [task_instruction + t for t in texts]

        embeddings = self.model.encode_multi_process(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            # convert_to_numpy=True,
            pool=self.pool
        )

        logger.info(f"Encoding completed, shape: {embeddings.shape}")
        return embeddings

    def _compute_similarities(self, query_embeddings: np.ndarray, corpus_embeddings: np.ndarray, query_chunk_size: int = 512, corpus_chunk_size: int = 512) -> np.ndarray:
        """
        Compute cosine similarities between query and corpus embeddings.

        Args:
            query_embeddings: Query embeddings, shape (n_queries, embedding_dim)
            corpus_embeddings: Corpus embeddings, shape (n_corpus, embedding_dim)
            query_chunk_size: Query chunk size, default 512
            corpus_chunk_size: Corpus chunk size, default 512

        Returns:
            Similarity matrix, shape (n_queries, n_corpus)
        """
        logger.info(f"Computing cosine similarities with query_chunk_size={query_chunk_size}, corpus_chunk_size={corpus_chunk_size}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        query_tensor = torch.from_numpy(query_embeddings).float()
        corpus_tensor = torch.from_numpy(corpus_embeddings).float()

        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
        corpus_tensor = torch.nn.functional.normalize(corpus_tensor, p=2, dim=1)

        n_queries = query_tensor.size(0)
        n_corpus = corpus_tensor.size(0)
        similarities = np.empty((n_queries, n_corpus), dtype=np.float32)

        query_chunk = query_chunk_size or n_queries
        corpus_chunk = corpus_chunk_size or n_corpus

        for q_start in range(0, n_queries, query_chunk):
            q_end = min(q_start + query_chunk, n_queries)
            q_batch = query_tensor[q_start:q_end].to(device)

            for c_start in range(0, n_corpus, corpus_chunk):
                c_end = min(c_start + corpus_chunk, n_corpus)
                c_batch = corpus_tensor[c_start:c_end].to(device)

                scores = torch.matmul(q_batch, c_batch.T).cpu().numpy()
                similarities[q_start:q_end, c_start:c_end] = scores

                del c_batch, scores

            del q_batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

        logger.info(
            f"Similarity matrix shape: {similarities.shape} (device={device}, "
        )
        return similarities

    def _prepare_trec_data(self, queries: List[Dict], corpus: List[Dict],
                           similarities: np.ndarray, qrels: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Prepare TREC evaluation data.

        Args:
            queries: Queries, list of dicts
            corpus: Corpus, list of dicts
            similarities: Similarity matrix, shape (n_queries, n_corpus)
            qrels: Qrels, list of dicts
            k: Top k, default 1000

        Returns:
            Run dict, dict of query_id to dict of corpus_id to score
            Qrel dict, dict of query_id to dict of corpus_id to relevance
        """
        logger.info(f"Preparing TREC evaluation data")

        run_dict = {}
        for idx, query_row in enumerate(tqdm(queries, total=len(queries))):
            query_id = query_row['_id']
            query_similarities = similarities[idx]

            top_k_indices = np.argsort(query_similarities)[::-1][:1000]

            run_dict[query_id] = {}
            for rank, corpus_idx in enumerate(top_k_indices):
                corpus_id = corpus[corpus_idx]['_id']
                score = float(query_similarities[corpus_idx])
                run_dict[query_id][corpus_id] = score

        qrel_dict = {}
        for qrel_row in qrels:
            query_id = qrel_row['query-id']
            corpus_id = qrel_row['corpus-id']
            relevance = int(qrel_row['score'])
            if query_id not in qrel_dict:
                qrel_dict[query_id] = {}
            qrel_dict[query_id][corpus_id] = relevance

        logger.info(f"Prepared run data for {len(run_dict)} queries")
        logger.info(f"Prepared qrel data for {len(qrel_dict)} queries")

        return run_dict, qrel_dict

    def _compute_metrics(self, run_dict: Dict, qrel_dict: Dict, measures: set[str]= {'map_cut', 'ndcg_cut', 'P', 'recall'}) -> Tuple[float, Dict[str, float]]:
        """
        Compute evaluation metrics.

        Args:
            run_dict: Run dict, dict of query_id to dict of corpus_id to score
            qrel_dict: Qrel dict, dict of query_id to dict of corpus_id to relevance
        Returns:
            Results, dict of query_id to dict of measure to score
        """
        logger.info(f"Computing evaluation metrics with measures={measures}")

        evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, measures)

        results = evaluator.evaluate(run_dict)

        return results

    def evaluate(self, qrels_path: str, queries_path: str, corpus_path: str, batch_size: int = 32, measures: set[str]= {'map_cut', 'ndcg_cut', 'P', 'recall'}, query_chunk_size: int = 512, corpus_chunk_size: int = 512) -> Dict[str, Dict]:
        """
        Evaluate retrieval performance.

        Args:
            qrels_path: Qrels path
            queries_path: Queries path
            corpus_path: Corpus path
            batch_size: Batch size, default 32
            measures: Measures, default {'map_cut', 'ndcg_cut', 'P', 'recall'}
        """
        logger.info("Starting retrieval evaluation")
        queries, corpus, qrels, qid2eval_res = self._load_data(queries_path, corpus_path, qrels_path)
        query_texts = [row['text'] for row in queries]
        corpus_texts = [row['text'] for row in corpus]

        query_embeddings = self._encode_texts(query_texts, batch_size * 2, is_query=True)
        corpus_embeddings = self._encode_texts(corpus_texts, batch_size, is_query=False)

        similarities = self._compute_similarities(query_embeddings, corpus_embeddings)

        run_dict, qrel_dict = self._prepare_trec_data(queries, corpus, similarities, qrels)

        query_results = self._compute_metrics(run_dict, qrel_dict)

        for qid, info in qid2eval_res.items():
            info["scores"] = query_results[qid]
        return qid2eval_res

def extract_lang(path):
    p = Path(path).parts[-3]
    if "-" in p and len(p.split("-")[0]) == 3:
        return p.split("-")[0]
    else:
        return "unk"

def extract_topic(path):
    return Path(path).parent.name

def extract_model_name(path):
    return Path(path).name


def compute_overall_scores(evaluation_results):
    overall_scores = defaultdict(list)
    for value in evaluation_results.values():
        scores = value["scores"]
        for key in scores:
            overall_scores[key].append(scores[key])
    
    for key in overall_scores:
        overall_scores[key] = np.mean(overall_scores[key])
    
    return overall_scores


if __name__ == '__main__':
    import argparse

    start_time = datetime.now()
    print("Start Time", start_time)

    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_path", required=True, type=str)
    parser.add_argument("--corpus_path", required=True, type=str)
    parser.add_argument("--qrels_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--batch_size", required=False, type=int, default=16)

    args = parser.parse_args()

    queries_path = args.queries_path
    corpus_path = args.corpus_path
    qrels_path = args.qrels_path
    model_path = args.model_path
    batch_size = args.batch_size

    query_lang = extract_lang(queries_path)
    corpus_lang = extract_lang(corpus_path)

    lang_pair = f"{query_lang}-{corpus_lang}"
    topic = extract_topic(queries_path)
    model_name = extract_model_name(model_path)
    save_path = f"evaluation_results/{model_name}/{lang_pair}/{topic}.json"
    print(f"{queries_path=}")
    print(f"{qrels_path=}")
    print(f"{corpus_path=}")
    print(f"{save_path=}")
    if os.path.exists(save_path):
        try:
            with open(save_path) as fp:
                _ = json.load(fp)
            print(f"{save_path} already exists, skip run!")
            sys.exit(0)
        except Exception as e:
            pass
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    evaluator = RetrievalEvaluator(model_path)
    # 0.6B: 80GB 16
    # 4B: 80GB 8
    # 8B: 80GB 4
    result = evaluator.evaluate(
        queries_path=queries_path,
        corpus_path=corpus_path,
        qrels_path=qrels_path,
        batch_size=batch_size
    )
    end_time = datetime.now()
    print("Finish Time：", end_time)
    print("Cost Time", (end_time - start_time).total_seconds(), "秒")
    all_info = {
        "domain": topic,
        "model_name": model_name,
        "model_path": model_path,
        "queries_path": queries_path,
        "corpus_path": corpus_path,
        "qrels_path": qrels_path,
        "overall_scores": compute_overall_scores(result),
        "evaluation_results": result
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_info, f, ensure_ascii=False, indent=4)