"""Filtered, type-constrained tail-ranking evaluation shared by every model.

For a test triple (h, r, t):
  1. score all entities as tails;
  2. mask entities that are not valid candidates (type constraint);
  3. mask every *other* known true tail of (h, r) (filtered setting);
  4. rank = 1 + #(strictly better) + 0.5 * #(tied with t, excluding t)   ("realistic" rank).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .dataset import Task

ScoreFn = Callable[[torch.Tensor], torch.Tensor]  # (B, 2) [h, r] -> (B, E) scores, higher = better


@torch.no_grad()
def ranks_for(task: Task, triples: np.ndarray, score_fn: ScoreFn, batch_size: int = 128, device: str = "cpu") -> np.ndarray:
    cand = torch.as_tensor(task.candidate_mask, device=device)
    out = []
    for i in range(0, len(triples), batch_size):
        b = torch.as_tensor(triples[i : i + batch_size], device=device)
        scores = score_fn(b[:, :2]).to(device).clone().float()
        scores[:, ~cand] = float("-inf")
        for j in range(len(b)):
            h, t = int(b[j, 0]), int(b[j, 2])
            other = [x for x in task.known_tails.get(h, ()) if x != t]
            if other:
                scores[j, other] = float("-inf")
        true = scores[torch.arange(len(b)), b[:, 2]].unsqueeze(1)
        better = (scores > true).sum(1)
        tied = (scores == true).sum(1) - 1
        out.append((1 + better + 0.5 * tied).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    if len(ranks) == 0:
        return {"n": 0, "mrr": float("nan"), "hits@1": float("nan"), "hits@3": float("nan"), "hits@10": float("nan"), "mean_rank": float("nan")}
    return {
        "n": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
    }


def evaluate(task: Task, score_fn: ScoreFn, split: str = "test", batch_size: int = 128, device: str = "cpu") -> dict:
    triples = task.test if split == "test" else task.valid
    return metrics_from_ranks(ranks_for(task, triples, score_fn, batch_size, device))
