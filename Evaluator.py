from __future__ import annotations

import math
import random

from itertools import chain
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

# Metric helpers & basics
MetricFn = Callable[[Sequence[bool], int], float]
SIM_TRIALS = 512  # <- used only if fallback sampling is requested

def _harmonic_dcg(length: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i in range(length)) # i + 2 because of 0-based counting


# Point metric functions
def _recall(preds: Sequence[bool], n_pos: int) -> float:
    return 0.0 if n_pos == 0 else sum(preds) / n_pos

def _precision(preds: Sequence[bool], _: int) -> float:
    return 0.0 if len(preds) == 0 else sum(preds) / len(preds)

def _f1(preds: Sequence[bool], n_pos: int) -> float:
    h = sum(preds)
    if h == 0:
        return 0.0
    return 2.0 * h / (len(preds) + n_pos)

def _average_precision(preds: Sequence[bool], n_pos: int) -> float:
    if n_pos == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for i, flag in enumerate(preds, 1):
        if flag:
            hits += 1
            ap += hits / i
    return ap / n_pos

def _mrr(preds: Sequence[bool], _: int) -> float:
    for i, flag in enumerate(preds, 1):
        if flag:
            return 1.0 / i
    return 0.0

def _ndcg(preds: Sequence[bool], n_pos: int) -> float:
    if n_pos == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(preds) if f) # i + 2 because of 0-based counting
    idcg = _harmonic_dcg(min(n_pos, len(preds)))
    return dcg / idcg if idcg else 0.0


METRICS: Dict[str, MetricFn] = {
    "recall": _recall,
    "precision": _precision,
    "f1": _f1,
    "map": _average_precision,
    "mrr": _mrr,
    "ndcg": _ndcg,
}


# Evaluator implementation
def _tie_groups(scores: Sequence[float], k: int) -> List[List[int]]:
    if k >= len(scores):
        unique = sorted(set(scores), reverse=True)
    else:
        thr = sorted(scores, reverse=True)[k - 1]
        unique = sorted({s for s in scores if s >= thr}, reverse=True)
    return [[i for i, s in enumerate(scores) if s == val] for val in unique]


@dataclass
class Evaluator:
    metric_fns: Dict[str, MetricFn] = field(default_factory=lambda: METRICS)

    # Public API
    def evaluate(
        self,
        is_relevant: List[List[bool]],
        pred_scores: List[List[float]],
        metrics: List[str] | None = None,
        k_list: List[int] | None = None,
        ) -> Dict[str, Dict[str, List[float]]]:

        if metrics is None:
            metrics = ["recall"]
        if k_list is None:
            k_list = [1, 3, 5, 10, 20, 50, 100]

        unknown = set(metrics) - self.metric_fns.keys()
        if unknown:
            raise KeyError(f"Unknown metrics: {', '.join(sorted(unknown))}")

        agg: Dict[str, Dict[str, List[float]]] = {
            m: {s: [] for s in ("tie-x", "expect", "maximum", "minimum")}
            for m in metrics
        }

        for k in k_list:
            acc = {m: {s: [] for s in agg[m]} for m in metrics}

            for rels, scores in zip(is_relevant, pred_scores):
                n_pos = sum(rels)
                groups = _tie_groups(scores, k)

                tie_preds = self._flatten(groups, rels)[:k]
                max_preds = self._optimistic(groups, rels, k)
                min_preds = self._pessimistic(groups, rels, k)

                for m in metrics:
                    fn = self.metric_fns[m]
                    acc[m]["tie-x"].append(fn(tie_preds, n_pos))
                    acc[m]["expect"].append(self._expected_metric(m, groups, rels, k))
                    acc[m]["maximum"].append(fn(max_preds, n_pos))
                    acc[m]["minimum"].append(fn(min_preds, n_pos))

            for m in metrics:
                for s, vals in acc[m].items():
                    agg[m][s].append(float(sum(vals) / len(vals)) if vals else 0.0)

        return agg

    # Expected metrics (analytic where possible)
    def _expected_metric(
        self,
        metric: str,
        groups: List[List[int]],
        rels: Sequence[bool],
        k: int,
    ) -> float:
        if metric == "recall":
            return self._expected_recall(groups, rels, k)
        if metric == "precision":
            return self._expected_precision(groups, rels, k)
        if metric == "f1":
            return self._expected_f1(groups, rels, k)
        if metric == "ndcg":
            return self._expected_ndcg(groups, rels, k)
        if metric == "map":
            return self._expected_ap(groups, rels, k)
        if metric == "mrr":
            return self._expected_rr(groups, rels, k)
        # fallback – not expected to be reached for built-ins
        return self._simulate_expectation(metric, groups, rels, k)

    # recall / precision / F1
    @staticmethod
    def _expected_hits(groups: List[List[int]], rels: Sequence[bool], k: int) -> float:
        """Expected number of relevant docs in top-k."""
        hits = 0.0
        rank = 0
        for grp in groups:
            p_rel = sum(rels[i] for i in grp) / len(grp)
            for _ in grp:
                rank += 1
                if rank > k:
                    return hits
                hits += p_rel
        return hits

    def _expected_precision(self, groups, rels, k):
        hits = self._expected_hits(groups, rels, k)
        return hits / k

    def _expected_recall(self, groups, rels, k):
        total_rel = sum(rels)
        if total_rel == 0:
            return 0.0
        return self._expected_hits(groups, rels, k) / total_rel

    def _expected_f1(self, groups, rels, k):
        n_pos = sum(rels)
        if n_pos == 0:
            return 0.0
        hits = self._expected_hits(groups, rels, k)
        return 2.0 * hits / (k + n_pos)

    # nDCG
    def _expected_ndcg(self, groups, rels, k):
        total_rel = sum(rels)
        if total_rel == 0:
            return 0.0
        dcg = 0.0
        rank = 0
        for grp in groups:
            p_rel = sum(rels[i] for i in grp) / len(grp)
            for _ in grp:
                rank += 1
                if rank > k:
                    break
                dcg += p_rel / math.log2(rank + 1)
            if rank >= k:
                break
        idcg = _harmonic_dcg(min(total_rel, k))
        return dcg / idcg if idcg else 0.0

    # Average Precision
    def _expected_ap(self, groups, rels, k):
        total_rel = sum(rels)
        if total_rel == 0:
            return 0.0

        numerator = 0.0
        rank = 0
        rel_seen_before_group = 0  # R_i in the paper
        for grp in groups:
            n_i = len(grp)
            x_i = sum(rels[i] for i in grp)
            if x_i == 0:
                rank += n_i
                continue
            frac_rel = x_i / n_i
            for offset in range(n_i):
                rank += 1
                if rank > k:
                    break
                # Probability position is relevant
                p_j_rel = frac_rel
                # Expected relevant before j *within the tie* conditioned on j being rel
                if n_i == 1:
                    inside = 0.0
                else:
                    inside = offset * (x_i - 1) / (n_i - 1)
                exp_rel_before_j = rel_seen_before_group + inside
                numerator += p_j_rel * (exp_rel_before_j + 1) / rank
            if rank >= k:
                break
            rel_seen_before_group += x_i
        return numerator / total_rel

    # Reciprocal Rank
    def _expected_rr(self, groups, rels, k):
        total_rel = sum(rels)
        if total_rel == 0:
            return 0.0

        prob_no_rel_yet = 1.0
        rr = 0.0
        rank = 0

        for grp in groups:
            n_i = len(grp)
            x_i = sum(rels[i] for i in grp)
            if x_i == 0:
                rank += n_i
                continue

            for offset in range(n_i):
                rank += 1
                if rank > k:
                    return rr

                p_all_prev = (math.comb(n_i - x_i, offset) /
                            math.comb(n_i, offset)) if offset else 1.0
                p_rel_here = x_i / (n_i - offset)          # ← 고친 부분
                prob_first = prob_no_rel_yet * p_all_prev * p_rel_here
                rr += prob_first / rank

            # 그룹에 정답이 하나라도 있으면 이후 P(no rel) = 0
            prob_no_rel_yet = 0.0
            if rank >= k:
                break
        return rr


    # Sampling fallback (not used for built-ins)                      
    def _simulate_expectation(self, metric, groups, rels, k, n_trials=SIM_TRIALS):
        fn = self.metric_fns[metric]
        rng = random.Random(42)
        total = 0.0
        for _ in range(n_trials):
            permuted = []
            for grp in groups:
                sels = grp[:]
                rng.shuffle(sels)
                permuted.extend(sels)
            preds = [rels[i] for i in permuted[:k]]
            total += fn(preds, sum(rels))
        return total / n_trials

    # Tie-handling helpers
    @staticmethod
    def _flatten(groups, rels):
        return [rels[i] for i in chain.from_iterable(groups)]

    @staticmethod
    def _optimistic(groups, rels, k):
        out = []
        for grp in groups:
            n_pos = sum(rels[i] for i in grp)
            out.extend([True] * n_pos + [False] * (len(grp) - n_pos))
            if len(out) >= k:
                break
        return out[:k]

    @staticmethod
    def _pessimistic(groups, rels, k):
        out = []
        for grp in groups:
            n_pos = sum(rels[i] for i in grp)
            out.extend([False] * (len(grp) - n_pos) + [True] * n_pos)
            if len(out) >= k:
                break
        return out[:k]
