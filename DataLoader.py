import mteb
import numpy as np
import os.path as osp

from copy import deepcopy
from tqdm.auto import tqdm
from functools import partial
from typing import Dict, List
from rank_bm25 import BM25Okapi
from collections import defaultdict
from copy import copy as shallowcopy
from datasets import Dataset, load_dataset, load_from_disk


DEV_MODE = False


class DataLoader:

    def __init__(self, cache_dir:str):
        self.cache_dir = cache_dir

    def load_dataset(
        self,
        dataset_name:str,
        bm25_top_n:int=1000
        ) -> Dict[str, List[str] | List[List[str]] | List[List[bool]]]:

        """
            MTEB (eng, v2)

            Reranking
            - AskUbuntuDupQuestions, s2s
            - MindSmallReranking, s2s

            Retrieval
            - ArguAna, s2p
            - ClimateFEVERHardNegatives, s2p
            - CQADupstackGamingRetrieval, s2p
            - CQADupstackUnixRetrieval, s2p
            - FEVERHardNegatives, s2p
            - FiQA2018, s2p
            - HotpotQAHardNegatives, s2p
            - SCIDOCS, s2p
            - Touche2020Retrieval.v3, s2p
            - TRECCOVID, s2p
        """

        if dataset_name == "MIRACLReranking":
            dict_of_list = self.load_MIRACLReranking()

        elif dataset_name == "MIRACLRetrieval":
            dict_of_list = self.load_MIRACLRetrieval()

        elif dataset_name == "AskUbuntuDupQuestions":
            dict_of_list = self.load_AskUbuntuDupQuestions(dataset_name)

        elif dataset_name == "MindSmallReranking":
            raise NotImplementedError(f"{dataset_name}")

        elif dataset_name in [
            "ArguAna", "ClimateFEVERHardNegatives",
            "CQADupstackGamingRetrieval", "CQADupstackUnixRetrieval",
            "FEVERHardNegatives", "FiQA2018",
            "HotpotQAHardNegatives", "SCIDOCS",
            "Touche2020Retrieval.v3", "TRECCOVID"]:
            dict_of_list = self.load_mteb_eng_v2_retrieval(dataset_name, bm25_top_n)

        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        if DEV_MODE:
            dict_of_list = {k: v[:10] for k, v in dict_of_list.items()}

        return dict_of_list

    def load_MIRACLReranking(self) -> Dict[str, List[str] | List[List[str]] | List[List[bool]]]:

        task = mteb.get_task("MIRACLReranking", languages=["eng"])
        task.load_data()
        dataset = task.dataset["en"]["dev"] # query:str, positive:List[str], negative:List[str], candidates:List[str]

        relevances = []
        include_indices = []
        for i, sample in tqdm(enumerate(dataset), desc="MIRACLReranking"):
            rels = [True if candidate in sample["positive"] else False
                for candidate in sample["candidates"]]
            relevances.append(rels)
            if any(rels): # any positives in candidates
                include_indices.append(i)

        queries = dataset.select(include_indices)["query"] # 799 -> 717
        candidates = dataset.select(include_indices)["candidates"] # 799 by 100 -> 717 by 100
        relevances = [rels for i, rels in enumerate(relevances) if i in include_indices] # 799 by 100 -> 717 by 100

        assert len(queries) == len(candidates) == len(relevances)
        assert all([len(cands) == 100 for cands in candidates])
        assert all([len(rels) == 100 for rels in relevances])

        return {
            "query": queries, # List[str]
            "candidates": candidates, # List[List[str]]
            "relevances": relevances} # List[List[bool]]

    def load_MIRACLRetrieval(self) -> Dict[str, List[str] | List[List[str]] | List[List[bool]]]:

        # Too slow!
        # task = mteb.get_task("MIRACLRetrieval", languages=["eng"])
        # task.load_data()
        # task.queries
        # task.corpus
        # task.relevant_docs

        # load # https://huggingface.co/datasets/mteb/MIRACLRetrieval
        load_miracl_retrieval = partial(
            load_dataset,
            path="mteb/MIRACLRetrieval",
            split="dev",
            cache_dir=self.cache_dir)
        dataset_queries = load_miracl_retrieval(name="en-queries") # 799
        dataset_corpus = load_miracl_retrieval(name="en-corpus") # 32893221
        dataset_rels = load_miracl_retrieval(name="en-qrels") # 8350

        # bm25_dataset_dir = f"{osp.join(*[self.cache_dir, 'tie_aware', data_name])}/bm25_{bm25_top_n}"

        # if osp.exists(bm25_dataset_dir):

        #     print("Loading BM25 candidates")
        #     dataset = load_from_disk(bm25_dataset_dir)
        #     return {
        #         "query": dataset["query"],
        #         "candidates": dataset["candidates"],
        #         "relevances": dataset["relevances"]}

        # else:

        print("Processing datasets...")
        query_ids = dataset_queries["_id"]
        corpus_ids = dataset_corpus["_id"]

        # assert all([x == 1 or x == 0 for x in dataset_rels["score"]]) # 1 or 0 # passed
        # assert len(set(query_ids)) == len(dataset_queries) # unique # passed
        # assert len(set(corpus_ids)) == len(dataset_corpus) # unique # passed
        # assert all([qid in query_ids for qid in tqdm(dataset_rels["query-id"], desc="assert")]) # passed
        # assert all([cid in corpus_ids for cid in tqdm(dataset_rels["corpus-id"], desc="assert")]) # passed

        queries = dataset_queries["text"]
        candidates = [dataset_corpus["text"]] * len(queries) # shallow copy

        query_id_to_i = {qid: i for i, qid in enumerate(query_ids)}
        corpus_id_to_i = {qid: i for i, qid in enumerate(corpus_ids)}
        relevances = [deepcopy([False] * len(dataset_corpus))] * len(queries) # init with deep copy
        for sample in dataset_rels:
            if sample["score"] == 1:
                query_i = query_id_to_i[sample["query-id"]]
                corpus_i = corpus_id_to_i[sample["corpus-id"]]
                relevances[query_i][corpus_i] = True

        return {
            "query": queries, # List[str]
            "candidates": candidates, # List[List[str]]
            "relevances": relevances} # List[List[bool]]

    def load_ArguAna(self) -> Dict[str, List[str] | List[List[str]] | List[List[bool]]]:

        # task = mteb.get_task("ArguAna", languages=["eng"])
        # task.load_data()
        # len(task.queries["test"]) # 1406
        # len(task.corpus["test"]) # 8674
        # len(task.relevant_docs["test"]) # 1406 # Dict[str, Dict[str, int]]
        # task.relevant_docs["test"]["test-environment-aeghhgwpe-pro02a"] >>> {'test-environment-aeghhgwpe-pro02b': 1}
        # task.corpus["test"]["test-environment-aeghhgwpe-pro02b"] >>> "animals environment ..."

        # https://huggingface.co/datasets/mteb/arguana
        dataset_queries = load_dataset("mteb/arguana", name="queries", split="queries") # 1406
        dataset_corpus = load_dataset("mteb/arguana", name="corpus", split="corpus") # 8674
        dataset_rels = load_dataset("mteb/arguana", name="default", split="test") # 1406 # represent relevances

        query_ids = dataset_queries["_id"]
        corpus_ids = dataset_corpus["_id"]

        assert all([x == 1 for x in dataset_rels["score"]])
        assert len(set(query_ids)) == len(dataset_queries) # unique
        assert len(set(corpus_ids)) == len(dataset_corpus) # unique
        assert all([qid in query_ids for qid in dataset_rels["query-id"]])
        # assert all([cid in corpus_ids for cid in dataset_rels["corpus-id"]]) # AssertionError!

        query_id_to_corpus_ids = defaultdict(list)
        for sample in dataset_rels:
            query_id_to_corpus_ids[sample["query-id"]].append(sample["corpus-id"])

        query_id_to_relevances = {}
        for query_id in query_ids:
            query_id_to_relevances[query_id] = [False] * len(dataset_corpus)
            for relevant_corpus_id in query_id_to_corpus_ids[query_id]:
                if relevant_corpus_id in corpus_ids: # because all([cid in corpus_ids for cid in dataset_rels["corpus-id"]]) is False
                    corpus_idx = corpus_ids.index(relevant_corpus_id)
                    query_id_to_relevances[query_id][corpus_idx] = True

        # num_relevants = []
        # for relevances in query_id_to_relevances.values():
        #     num_relevants.append(len([x for x in relevances if x is True]))
        # print(Counter(num_relevants)) # {1: 1401, 0: 5}

        queries = dataset_queries["text"]
        candidates = [f"{sample['title']}\n{sample['text']}" for sample in dataset_corpus] # https://github.com/embeddings-benchmark/mteb/blob/17be7e548dbd3080e9dcc1abdc509d6762ccf1b6/mteb/models/bm25.py#L71
        relevances = list(query_id_to_relevances.values())

        # drop the queries what have no relevant corpus
        ignore_indices = []
        for i, query_id in enumerate(query_ids):
            if not any(query_id_to_relevances[query_id]):
                ignore_indices.append(i)

        queries = [x for i, x in enumerate(queries) if i not in ignore_indices] # 1406 -> 1401
        candidates = [shallowcopy(candidates)] * len(queries) # 1401 by 8674 # NOTE: stays shallow until Dataset.from_dict
        relevances = [x for i, x in enumerate(relevances) if i not in ignore_indices] # 1406 by 8674 -> 1401 by 8674
        assert len(queries) == len(relevances)
        assert all([len(cands) == len(rels) for cands, rels in zip(candidates, relevances)])

        return {
            "query": queries, # List[str]
            "candidates": candidates, # List[List[str]]
            "relevances": relevances} # List[List[bool]]
    
    def load_AskUbuntuDupQuestions(
        self,
        data_name:str
        ) -> Dict[str, List[str] | List[List[str]] | List[List[bool]]]:

        task = mteb.get_task(data_name, languages=["eng"])
        task.load_data()
        
        queries, candidates, relevances = [], [], []
        for sample in task.dataset["test"]:

            cands = sample["positive"] + sample["negative"]
            rels = [True] * len(sample["positive"]) + [False] * len(sample["negative"])

            queries.append(sample["query"])
            candidates.append(cands)
            relevances.append(rels)

        return {
            "query": queries, # List[str]
            "candidates": candidates, # List[List[str]]
            "relevances": relevances} # List[List[bool]]

    def load_mteb_eng_v2_retrieval(
        self,
        data_name:str,
        bm25_top_n:int=1000
        ) -> Dict[str, List[str] | List[List[str]] | List[List[bool]]]:

        bm25_dataset_dir = f"{osp.join(*[self.cache_dir, 'tie_aware', data_name])}/bm25_{bm25_top_n}"

        if osp.exists(bm25_dataset_dir):

            print("Loading BM25 candidates")
            dataset = load_from_disk(bm25_dataset_dir)
            return {
                "query": dataset["query"],
                "candidates": dataset["candidates"],
                "relevances": dataset["relevances"]}

        else:
        # if True: # HACK

            task = mteb.get_task(data_name, languages=["eng"])
            task.load_data()

            query_dict = task.queries["test"] # Dict[str, str]
            corpus_dict = task.corpus["test"] # Dict[str, str]
            relevance_dict = task.relevant_docs["test"] # Dict[str, Dict[str, int]]

            corpus_id_to_i = {corpus_id: i for i, corpus_id in enumerate(corpus_dict.keys())}
            assert set(query_dict.keys()) == set(relevance_dict.keys())
            relevance_dict = {qid: relevance_dict[qid] for qid in query_dict.keys()}
            assert all([q == r for q, r in zip(query_dict.keys(), relevance_dict.keys())])

            relevances = []
            for i_query, corpus_id_to_label in enumerate(relevance_dict.values()):
                rels = [False] * len(corpus_dict) # init with all False
                for corpus_id, label in corpus_id_to_label.items():
                    if label == 1 and corpus_id in corpus_id_to_i:
                        i_corpus = corpus_id_to_i[corpus_id]
                        rels[i_corpus] = True
                relevances.append(rels)

            queries = list(query_dict.values())
            candidates = [list(corpus_dict.values())] * len(queries) # shallow copy
            # print(f"# of queries = {len(queries)}")
            # print(f"# of candidates = {len(candidates[0])}")
            # query_lens = [len(x) for x in queries]
            # candidate_lens = [len(x) for x in candidates[0]]
            # print(f"Avg. query length = {sum(query_lens) / len(query_lens):.2f}")
            # print(f"Avg. candidate length = {sum(candidate_lens) / len(candidate_lens):.2f}")
            # breakpoint()

            # return {
            #     "query": queries, # List[str]
            #     "candidates": candidates, # List[List[str]]
            #     "relevances": relevances} # List[List[bool]]

            # bm25
            if DEV_MODE:
                queries = queries[:2]
                candidates = candidates[:2]
                relevances = relevances[:2]

            candidates_bm25, relevances_bm25 = [], []

            cands = candidates[0] # shared candidates
            tokenized_corpus = [x.split() for x in cands]
            bm25 = BM25Okapi(tokenized_corpus)

            for i_query in tqdm(range(len(queries)), desc="bm25"):

                query = queries[i_query]
                tokenized_query = query.split()
                cands_bm25 = bm25.get_top_n(tokenized_query, cands, n=bm25_top_n)
                rels = relevances[i_query]
                rels_bm25 = [rels[cands.index(cand_bm25)] for cand_bm25 in cands_bm25]

                # query = queries[i_query]
                # tokenized_query = query.split()
                # rels = relevances[i_query]
                # scores = bm25.get_scores(tokenized_query)
                # if bm25_top_n < len(scores):
                #     top_idx_part = np.argpartition(scores, -bm25_top_n)[-bm25_top_n:]
                # else:
                #     top_idx_part = np.arange(len(scores))
                # top_idx = top_idx_part[np.argsort(scores[top_idx_part])[::-1]]
                # cands_bm25 = [cands[i] for i in top_idx]
                # rels_bm25  = [rels[i]  for i in top_idx]

                candidates_bm25.append(cands_bm25)
                relevances_bm25.append(rels_bm25)

            dataset = Dataset.from_dict({
                "query": queries, # List[str]
                "candidates": candidates_bm25, # List[List[str]]
                "relevances": relevances_bm25}) # List[List[bool]]

            dataset.save_to_disk(bm25_dataset_dir)
            dataset = load_from_disk(bm25_dataset_dir)
            return {
                "query": dataset["query"],
                "candidates": dataset["candidates"],
                "relevances": dataset["relevances"]}
            # # This makes inference slow!
            # return {
            #     "query": queries, # List[str]
            #     "candidates": candidates, # List[List[str]]
            #     "relevances": relevances} # List[List[bool]]