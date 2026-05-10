from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from pandas.api.types import is_categorical_dtype, is_string_dtype
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F


def set_seed(seed: int = 0, deterministic: bool = False) -> torch.device:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def clean_gene_names(var_names: Iterable[str]) -> np.ndarray:
    out = []
    for g in var_names:
        g = str(g)
        g = g.replace(".0", "").replace(".1", "").replace(".2", "")
        out.append(g)
    return np.array(out, dtype=str)


def build_knn_edges(coords, k: int = 2) -> np.ndarray:
    coords = np.asarray(coords)
    n = coords.shape[0]
    if n <= 1:
        return np.empty((0, 2), dtype=np.int64)

    k_eff = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nbrs.fit(coords)
    idx = nbrs.kneighbors(return_distance=False)

    src = np.repeat(np.arange(n), idx.shape[1] - 1)
    dst = idx[:, 1:].reshape(-1)
    edges = np.stack([src, dst], axis=1).astype(np.int64)
    rev = edges[:, ::-1]
    return np.concatenate([edges, rev], axis=0)


def graph_smoothness(z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.shape[0] == 0:
        return torch.tensor(0.0, device=z.device)
    zi = z[edges[:, 0]]
    zj = z[edges[:, 1]]
    return ((zi - zj) ** 2).mean()


def freeze_module(module, freeze: bool = True) -> None:
    for p in module.parameters():
        p.requires_grad = not freeze


def sanitize_dataframe_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if is_string_dtype(s) or is_categorical_dtype(s):
            out[col] = s.astype(object)
    return out


def make_safe_var(index, source_labels, imputed_flags) -> pd.DataFrame:
    index = np.asarray(index, dtype=str)
    var = pd.DataFrame(index=pd.Index(index, name=None))
    var["source"] = np.asarray(source_labels, dtype=object)
    var["imputed"] = np.asarray(imputed_flags, dtype=bool)
    return var


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights.unsqueeze(0)).mean()


def encode_labels(values):
    labels = pd.Series(values).astype(str).fillna("<UNK>").tolist()
    uniq = sorted(set(labels))
    if len(uniq) == 1:
        uniq = uniq + ["<UNK>"]
    mapping = {lab: i for i, lab in enumerate(uniq)}
    codes = np.array([mapping.get(x, mapping["<UNK>"]) for x in labels], dtype=np.int64)
    return codes, uniq


def make_class_weights(codes, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(codes, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def detect_layer56_binary(values) -> np.ndarray:
    out = []
    for v in pd.Series(values).astype(str).fillna("<UNK>").tolist():
        s = v.strip().lower()
        s = s.replace(" ", "").replace("_", "").replace("-", "")
        s = s.replace(".0", "")
        s = re.sub(r"[^a-z0-9/]+", "", s)

        is5 = s == "5" or s.startswith("l5") or s.startswith("layer5") or s.startswith("lamina5")
        is6 = s == "6" or s.startswith("l6") or s.startswith("layer6") or s.startswith("lamina6")

        if is5 and not is6:
            out.append(1)
        elif is6 and not is5:
            out.append(0)
        else:
            out.append(-1)
    return np.array(out, dtype=np.int64)


def layer56_margin_loss(z: torch.Tensor, labels: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    valid = labels >= 0
    if valid.sum().item() == 0:
        return torch.tensor(0.0, device=z.device)

    labels = labels[valid]
    z = F.normalize(z[valid], dim=1)
    if (labels == 0).sum().item() < 2 or (labels == 1).sum().item() < 2:
        return torch.tensor(0.0, device=z.device)

    c0 = z[labels == 0].mean(dim=0)
    c1 = z[labels == 1].mean(dim=0)
    cos_sim = F.cosine_similarity(c0.unsqueeze(0), c1.unsqueeze(0)).squeeze(0)
    return F.relu(cos_sim - margin)
