"""Link-prediction models: PyKEEN embeddings (TransE / RotatE / ComplEx / ...), a PyG R-GCN, a popularity baseline."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .dataset import Task

log = logging.getLogger(__name__)


@dataclass
class Trained:
    name: str
    score_fn: Callable[[torch.Tensor], torch.Tensor]
    train_seconds: float
    final_loss: float | None = None


# ----------------------------------------------------------------------------- PyKEEN embeddings
def train_pykeen(task: Task, model_name: str, dim: int, epochs: int, batch_size: int, seed: int, device: str = "cpu",
                 lr: float = 0.01) -> Trained:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    def tf(arr: np.ndarray) -> TriplesFactory:
        return TriplesFactory(mapped_triples=torch.as_tensor(arr), entity_to_id=task.entity_to_id,
                              relation_to_id=task.relation_to_id, create_inverse_triples=False)

    # PyKEEN's pipeline insists on a testing set and evaluates once at the end; hand it a tiny slice so
    # that cost is negligible. The real evaluation is done by our own evaluator (ml/evaluate.py).
    tiny = task.valid[: max(1, min(len(task.valid), 20))]
    t0 = time.time()
    res = pipeline(
        training=tf(task.train), validation=tf(tiny), testing=tf(tiny),
        model=model_name, model_kwargs=dict(embedding_dim=dim),
        # PyKEEN's default Adam lr (1e-3) badly under-trains at the step counts used here (see tests/test_ml.py).
        optimizer_kwargs=dict(lr=lr),
        training_loop="slcwa",
        training_kwargs=dict(num_epochs=epochs, batch_size=batch_size, use_tqdm=False),
        random_seed=seed, device=device,
    )
    model = res.model.to(device).eval()

    def score(hr: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.score_t(hr_batch=hr.to(device))

    return Trained(f"pykeen:{model_name}", score, time.time() - t0, float(res.losses[-1]) if res.losses else None)


# ----------------------------------------------------------------------------- PyG R-GCN
def _make_rgcn(num_nodes: int, num_rel: int, dim: int):
    import torch.nn as nn
    from torch_geometric.nn import RGCNConv

    class RGCNLinkPredictor(nn.Module):
        """2-layer R-GCN encoder (basis decomposition) + DistMult decoder.

        Message passing uses each relation and its inverse (relation id + num_rel); the decoder only
        scores forward relations."""

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(num_nodes, dim)
            nb = min(8, 2 * num_rel)
            self.conv1 = RGCNConv(dim, dim, num_relations=2 * num_rel, num_bases=nb)
            self.conv2 = RGCNConv(dim, dim, num_relations=2 * num_rel, num_bases=nb)
            self.rel = nn.Parameter(torch.empty(num_rel, dim))
            nn.init.xavier_uniform_(self.rel)
            nn.init.xavier_uniform_(self.emb.weight)

        def encode(self, edge_index, edge_type):
            x = self.emb.weight
            x = torch.relu(self.conv1(x, edge_index, edge_type))
            return self.conv2(x, edge_index, edge_type)

        def score(self, z, h, r, t):
            return (z[h] * self.rel[r] * z[t]).sum(-1)

        def score_all_tails(self, z, h, r):
            return (z[h] * self.rel[r]) @ z.T

    return RGCNLinkPredictor()


def _message_graph(triples: np.ndarray, num_rel: int, device: str):
    h = torch.as_tensor(triples[:, 0], device=device)
    r = torch.as_tensor(triples[:, 1], device=device)
    t = torch.as_tensor(triples[:, 2], device=device)
    ei = torch.stack([torch.cat([h, t]), torch.cat([t, h])])
    et = torch.cat([r, r + num_rel])
    return ei, et


def train_rgcn(task: Task, dim: int, epochs: int, seed: int, lr: float = 0.01, device: str = "cpu") -> Trained:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _make_rgcn(task.num_entities, task.num_relations, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train = task.train
    full_ei, full_et = _message_graph(train, task.num_relations, device)
    t0 = time.time()
    last = None
    for ep in range(epochs):
        model.train()
        # Standard R-GCN link-prediction trick: split the training edges into a message-passing half and a
        # supervision half each epoch, so a positive edge is never also used to compute its own embedding.
        perm = rng.permutation(len(train))
        half = len(train) // 2
        mp, sup = train[perm[:half]], train[perm[half:]]
        ei, et = _message_graph(mp, task.num_relations, device)
        z = model.encode(ei, et)
        s = torch.as_tensor(sup, device=device)
        pos = model.score(z, s[:, 0], s[:, 1], s[:, 2])
        neg_t = torch.randint(0, task.num_entities, (len(s),), device=device)
        neg = model.score(z, s[:, 0], s[:, 1], neg_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    model.eval()
    with torch.no_grad():
        z_full = model.encode(full_ei, full_et)

    def score(hr: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hr = hr.to(device)
            return model.score_all_tails(z_full, hr[:, 0], hr[:, 1])

    return Trained("pyg:RGCN-DistMult", score, time.time() - t0, last)


# ----------------------------------------------------------------------------- baseline
def popularity_baseline(task: Task) -> Trained:
    """Score a tail by how often it is a tail of the target relation in the training graph (head-independent)."""
    r = task.target_rel_id
    counts = np.bincount(task.train[task.train[:, 1] == r][:, 2], minlength=task.num_entities).astype(np.float32)
    vec = torch.as_tensor(counts)

    def score(hr: torch.Tensor) -> torch.Tensor:
        return vec.unsqueeze(0).expand(len(hr), -1).clone()

    return Trained("baseline:popularity", score, 0.0)
