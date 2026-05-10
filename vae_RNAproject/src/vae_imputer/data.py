from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import anndata as ad
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import (
    clean_gene_names,
    encode_labels,
    make_class_weights,
    detect_layer56_binary,
    to_dense,
)


@dataclass
class PipelineData:
    adata_sp: ad.AnnData
    adata_sc: ad.AnnData
    adata_sp_shared: ad.AnnData
    adata_sc_shared: ad.AnnData
    adata_sc_target: ad.AnnData
    shared_genes: np.ndarray
    target_genes: np.ndarray
    target_missing_in_sp: np.ndarray
    cond_sc: np.ndarray
    cond_sp: np.ndarray
    subclass_vocab: list
    layer_codes: Optional[np.ndarray]
    layer_vocab: Optional[list]
    layer_class_weights: Optional[torch.Tensor]
    layer56_codes: Optional[np.ndarray]
    layer56_enabled: bool
    layer56_class_weights: Optional[torch.Tensor]
    X_sc_shared: np.ndarray
    X_sc_target: np.ndarray
    X_sp_shared: np.ndarray
    X_sp_all: np.ndarray
    target_weights: torch.Tensor


class SCPairedDataset(Dataset):
    def __init__(self, X_shared, X_target, subclass):
        self.X_shared = torch.tensor(X_shared, dtype=torch.float32)
        self.X_target = torch.tensor(X_target, dtype=torch.float32)
        self.subclass = torch.tensor(subclass, dtype=torch.long)

    def __len__(self):
        return self.X_shared.shape[0]

    def __getitem__(self, idx):
        return self.X_shared[idx], self.X_target[idx], self.subclass[idx]


@dataclass
class DataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader


def load_adata(sp_path: str, sc_path: str):
    adata_sp = sc.read_h5ad(sp_path)
    adata_sc = sc.read_h5ad(sc_path)
    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()
    adata_sc.var_names = clean_gene_names(adata_sc.var_names)
    adata_sp.var_names = clean_gene_names(adata_sp.var_names)
    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()
    return adata_sp, adata_sc


def prepare_data(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    device: torch.device,
    st_hvg_n: int = 3000,
    sc_hvg_n: int = 5000,
    layer_marker_genes: Optional[list[str]] = None,
):
    layer_marker_genes = layer_marker_genes or []

    try:
        sc.pp.highly_variable_genes(adata_sp, n_top_genes=st_hvg_n, flavor="seurat_v3")
    except Exception:
        sc.pp.highly_variable_genes(adata_sp, n_top_genes=st_hvg_n, flavor="seurat")
    st_hvg_genes = np.array(adata_sp.var_names[adata_sp.var["highly_variable"]], dtype=str)

    try:
        sc.pp.highly_variable_genes(adata_sc, n_top_genes=sc_hvg_n, flavor="seurat_v3")
    except Exception:
        sc.pp.highly_variable_genes(adata_sc, n_top_genes=sc_hvg_n, flavor="seurat")
    sc_hvg_genes = np.array(adata_sc.var_names[adata_sc.var["highly_variable"]], dtype=str)

    marker_genes = np.array(
        [g for g in layer_marker_genes if g in set(adata_sp.var_names) or g in set(adata_sc.var_names)],
        dtype=str,
    )

    for adata in [adata_sc, adata_sp]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    shared_genes = np.intersect1d(adata_sp.var_names, adata_sc.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes found. Please check gene naming between scRNA and spatial data.")

    target_union = []
    target_union.extend(list(marker_genes))
    target_union.extend(list(st_hvg_genes))
    target_union.extend(list(sc_hvg_genes))
    target_genes = np.array(sorted(set(target_union)), dtype=str)
    target_genes = np.array([g for g in target_genes if g in adata_sc.var_names], dtype=str)
    if len(target_genes) == 0:
        raise ValueError("No target genes found after filtering.")

    target_missing_in_sp = np.array([g for g in target_genes if g not in set(adata_sp.var_names)], dtype=str)

    adata_sp_shared = adata_sp[:, shared_genes].copy()
    adata_sc_shared = adata_sc[:, shared_genes].copy()
    adata_sc_target = adata_sc[:, target_genes].copy()

    sc_codes, sp_codes, subclass_vocab = make_shared_category_codes(adata_sc.obs, adata_sp.obs, col="Subclass")

    layer_col = None
    for c in ["layer", "Layer", "layers", "layer_label", "lamina", "Lamina", "LAYER"]:
        if c in adata_sp.obs.columns:
            vals = adata_sp.obs[c].astype(str).tolist()
            if len(sorted(set(vals))) >= 2:
                layer_col = c
                break

    if layer_col is not None:
        layer_codes, layer_vocab = encode_labels(adata_sp.obs[layer_col])
        layer_class_weights = make_class_weights(layer_codes, len(layer_vocab), device)
        layer56_codes = detect_layer56_binary(adata_sp.obs[layer_col])
        n_layer5 = int((layer56_codes == 1).sum())
        n_layer6 = int((layer56_codes == 0).sum())
        layer56_enabled = n_layer5 >= 5 and n_layer6 >= 5
        layer56_class_weights = make_class_weights(layer56_codes[layer56_codes >= 0], 2, device) if layer56_enabled else None
    else:
        layer_codes = None
        layer_vocab = None
        layer_class_weights = None
        layer56_codes = None
        layer56_enabled = False
        layer56_class_weights = None

    X_sc_shared = to_dense(adata_sc_shared.X).astype(np.float32, copy=False)
    X_sc_target = to_dense(adata_sc_target.X).astype(np.float32, copy=False)
    X_sp_shared = to_dense(adata_sp_shared.X).astype(np.float32, copy=False)
    X_sp_all = to_dense(adata_sp.X).astype(np.float32, copy=False)

    st_hvg_set = set(st_hvg_genes.tolist())
    sc_hvg_set = set(sc_hvg_genes.tolist())
    marker_set = set(marker_genes.tolist())
    target_weights = []
    for g in target_genes:
        w = 1.0
        if g in marker_set:
            w *= 4.0
        if g in st_hvg_set:
            w *= 2.5
        if g in sc_hvg_set and g not in st_hvg_set:
            w *= 0.4
        if g in st_hvg_set and g in sc_hvg_set:
            w *= 1.3
        target_weights.append(w)
    target_weights = torch.tensor(np.array(target_weights, dtype=np.float32), device=device)

    return PipelineData(
        adata_sp=adata_sp,
        adata_sc=adata_sc,
        adata_sp_shared=adata_sp_shared,
        adata_sc_shared=adata_sc_shared,
        adata_sc_target=adata_sc_target,
        shared_genes=shared_genes,
        target_genes=target_genes,
        target_missing_in_sp=target_missing_in_sp,
        cond_sc=sc_codes,
        cond_sp=sp_codes,
        subclass_vocab=subclass_vocab,
        layer_codes=layer_codes,
        layer_vocab=layer_vocab,
        layer_class_weights=layer_class_weights,
        layer56_codes=layer56_codes,
        layer56_enabled=layer56_enabled,
        layer56_class_weights=layer56_class_weights,
        X_sc_shared=X_sc_shared,
        X_sc_target=X_sc_target,
        X_sp_shared=X_sp_shared,
        X_sp_all=X_sp_all,
        target_weights=target_weights,
    )


def make_shared_category_codes(sc_obs, sp_obs, col="Subclass"):
    sc_labels = sc_obs[col].astype(str).tolist()
    sp_labels = sp_obs[col].astype(str).tolist()
    uniq = sorted(set(sc_labels) | set(sp_labels))
    if "<UNK>" not in uniq:
        uniq = uniq + ["<UNK>"]
    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    unk_idx = label_to_idx["<UNK>"]
    sc_codes = np.array([label_to_idx.get(x, unk_idx) for x in sc_labels], dtype=np.int64)
    sp_codes = np.array([label_to_idx.get(x, unk_idx) for x in sp_labels], dtype=np.int64)
    return sc_codes, sp_codes, uniq


def make_loaders(X_sc_shared, X_sc_target, cond_sc, batch_size: int = 256, val_ratio: float = 0.1, seed: int = 0):
    n_sc = X_sc_shared.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sc)
    val_size = max(1, int(val_ratio * n_sc))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    train_loader = DataLoader(
        SCPairedDataset(X_sc_shared[train_idx], X_sc_target[train_idx], cond_sc[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SCPairedDataset(X_sc_shared[val_idx], X_sc_target[val_idx], cond_sc[val_idx]),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return DataLoaders(train_loader=train_loader, val_loader=val_loader)
