from __future__ import annotations

import numpy as np
import anndata as ad
import torch

from .data import load_adata, prepare_data, make_loaders
from .model import STTopGeneImputer
from .training import pretrain_sc, finetune_st
from .utils import sanitize_dataframe_for_h5ad, make_safe_var


def run_pipeline(
    base_path: str,
    sp_rel_path: str,
    sc_rel_path: str,
    output_file: str,
    layer_marker_genes=None,
    seed: int = 0,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata_sp, adata_sc = load_adata(
        sp_path=f"{base_path}/{sp_rel_path}",
        sc_path=f"{base_path}/{sc_rel_path}",
    )
    pdata = prepare_data(adata_sp, adata_sc, device=device, layer_marker_genes=layer_marker_genes or [])
    loaders = make_loaders(pdata.X_sc_shared, pdata.X_sc_target, pdata.cond_sc, seed=seed)

    model = STTopGeneImputer(
        n_shared=len(pdata.shared_genes),
        n_target=len(pdata.target_genes),
        n_subclass=len(pdata.subclass_vocab),
        n_layer=len(pdata.layer_vocab) if pdata.layer_vocab is not None else 0,
        use_layer56=pdata.layer56_enabled,
        z_dim=64,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model, _ = pretrain_sc(
        model,
        loaders.train_loader,
        loaders.val_loader,
        optimizer,
        device=device,
        target_weights=pdata.target_weights,
    )

    layer_codes_t = torch.tensor(pdata.layer_codes, dtype=torch.long, device=device) if pdata.layer_codes is not None else None
    layer56_codes_t = torch.tensor(pdata.layer56_codes, dtype=torch.long, device=device) if pdata.layer56_codes is not None and pdata.layer56_enabled else None
    model, _ = finetune_st(
        model,
        pdata.adata_sp,
        pdata.X_sp_shared,
        layer_codes_t,
        layer56_codes_t,
        pdata.layer_class_weights,
        pdata.layer56_class_weights,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        mu_sp, _ = model.encode(torch.tensor(pdata.X_sp_shared, dtype=torch.float32, device=device))
        target_pred_sp = model.decode_target(mu_sp)
        if target_pred_sp is None:
            target_pred_sp = np.zeros((pdata.adata_sp.n_obs, 0), dtype=np.float32)
        else:
            target_pred_sp = target_pred_sp.detach().cpu().numpy()
    target_pred_sp = np.maximum(target_pred_sp, 0.0).astype(np.float32)

    sp_gene_set = set(pdata.adata_sp.var_names)
    extra_target_genes = np.array([g for g in pdata.target_genes if g not in sp_gene_set], dtype=str)
    extra_target_idx = np.array([i for i, g in enumerate(pdata.target_genes) if g not in sp_gene_set], dtype=np.int64)
    extra_pred_sp = target_pred_sp[:, extra_target_idx].astype(np.float32) if len(extra_target_idx) > 0 else np.zeros((pdata.adata_sp.n_obs, 0), dtype=np.float32)
    X_full_sp = np.concatenate([pdata.X_sp_all, extra_pred_sp], axis=1).astype(np.float32)

    obs_save = sanitize_dataframe_for_h5ad(pdata.adata_sp.obs.copy())
    adata_full_sp = ad.AnnData(
        X=X_full_sp,
        obs=obs_save,
        uns=pdata.adata_sp.uns.copy(),
        obsm=pdata.adata_sp.obsm.copy(),
    )
    all_gene_names = np.concatenate([pdata.adata_sp.var_names.to_numpy(), extra_target_genes]).astype(str)
    source_labels = np.array(["ST_observed"] * pdata.adata_sp.n_vars + ["imputed_target"] * len(extra_target_genes), dtype=object)
    imputed_flags = np.array([False] * pdata.adata_sp.n_vars + [True] * len(extra_target_genes), dtype=bool)
    adata_full_sp.var = make_safe_var(all_gene_names, source_labels, imputed_flags)
    adata_full_sp.obsm["pred_targets"] = target_pred_sp.astype(np.float32)
    adata_full_sp.uns["target_genes"] = [str(g) for g in pdata.target_genes]
    adata_full_sp.write(output_file)

    return adata_full_sp
