from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import PipelineConfig
from models import DS, Projector, VAE


def set_seed(seed=0, deterministic=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(False)


def dense(x):
    return x.toarray() if not isinstance(x, np.ndarray) else x


def valid_subclass(sc_sub, atac_sub, sp_sub):
    sc_set = set(pd.Series(sc_sub).astype(str))
    atac_set = set(pd.Series(atac_sub).astype(str))
    sp_set = set(pd.Series(sp_sub).astype(str))

    if not sc_set.issubset(sp_set):
        return False
    if not atac_set.issubset(sp_set):
        return False
    if not atac_set.issubset(sc_set.union(sp_set)):
        return False
    return True


def build_shared_class_codes(adata_sp, adata_sc, adata_atac):
    all_labels = pd.Index(
        pd.concat(
            [
                adata_sp.obs["Subclass"].astype(str),
                adata_sc.obs["Subclass"].astype(str),
                adata_atac.obs["Subclass"].astype(str),
            ],
            axis=0,
        ).unique()
    ).sort_values()

    def encode(adata):
        return torch.tensor(
            pd.Categorical(
                adata.obs["Subclass"].astype(str),
                categories=all_labels
            ).codes
        ).long()

    return all_labels.tolist(), encode(adata_sp), encode(adata_sc), encode(adata_atac)


def cov(x):
    if x.shape[0] <= 1:
        return torch.zeros((x.shape[1], x.shape[1]), device=x.device, dtype=x.dtype)
    x = x - x.mean(0, keepdim=True)
    return x.T @ x / (x.shape[0] - 1)


def align(z1, c1, z2, c2):
    loss = 0.0
    for k in torch.unique(c1):
        if (c2 == k).sum() == 0:
            continue
        loss = loss + F.mse_loss(z1[c1 == k].mean(0), z2[c2 == k].mean(0))
    return loss


def train_vae(model, loader, opt, device, epochs=30, kl_weight=1e-4):
    model.train()
    for e in range(epochs):
        tot = 0.0
        for x, c in loader:
            x = x.to(device)
            c = c.to(device)

            r, mu, lv, z = model(x, c)

            rec = F.poisson_nll_loss(r, x, log_input=False)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = rec + kl_weight * kl

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()

        if e % 5 == 0:
            print("epoch", e, tot)


@torch.no_grad()
def encode_mu(model, X, c, device):
    x = torch.tensor(X).float().to(device)
    c = c.to(device)
    mu, _ = model.encode(x, c)
    return mu


def train_projector(proj, mu_sp, c_sp, mu_sc, c_sc, mu_atac, c_atac, opt, device, epochs=200):
    proj.train()
    for e in range(epochs):
        z_sp = proj(mu_sp)
        z_sc = proj(mu_sc)
        z_atac = proj(mu_atac)

        cov_loss = (
            F.mse_loss(cov(z_sp), cov(z_sc))
            + F.mse_loss(cov(z_sp), cov(z_atac))
            + F.mse_loss(cov(z_sc), cov(z_atac))
        )

        cls_loss = (
            align(z_sp, c_sp, z_sc, c_sc)
            + align(z_sp, c_sp, z_atac, c_atac)
            + align(z_sc, c_sc, z_atac, c_atac)
        )

        loss = cov_loss + cls_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if e % 20 == 0:
            print("align", e, loss.item())


def load_triplet(sp_path, sc_path, atac_path):
    adata_sp = sc.read_h5ad(sp_path)
    adata_sc = sc.read_h5ad(sc_path)
    adata_atac = sc.read_h5ad(atac_path)
    return adata_sp, adata_sc, adata_atac


def preprocess_triplet(adata_sp, adata_sc, adata_atac, hvg_top_genes=3000):
    adata_sp = adata_sp.copy()
    adata_sc = adata_sc.copy()
    adata_atac = adata_atac.copy()

    sc.pp.highly_variable_genes(adata_sc, n_top_genes=hvg_top_genes, flavor="seurat_v3")
    adata_sc = adata_sc[:, adata_sc.var["highly_variable"]].copy()

    sc.pp.highly_variable_genes(adata_atac, n_top_genes=hvg_top_genes, flavor="seurat_v3")
    adata_atac = adata_atac[:, adata_atac.var["highly_variable"]].copy()

    shared_genes = np.intersect1d(adata_sp.var_names, adata_sc.var_names)
    adata_sp = adata_sp[:, shared_genes].copy()

    for adata in [adata_sp, adata_sc, adata_atac]:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata_sp, adata_sc, adata_atac


def make_loaders(adata_sp, adata_sc, adata_atac, batch_size=256):
    _, c_sp, c_sc, c_atac = build_shared_class_codes(adata_sp, adata_sc, adata_atac)

    X_sp = dense(adata_sp.X)
    X_sc = dense(adata_sc.X)
    X_atac = dense(adata_atac.X)

    loader_sp = DataLoader(DS(X_sp, c_sp), batch_size=batch_size, shuffle=True)
    loader_sc = DataLoader(DS(X_sc, c_sc), batch_size=batch_size, shuffle=True)
    loader_atac = DataLoader(DS(X_atac, c_atac), batch_size=batch_size, shuffle=True)

    return (X_sp, X_sc, X_atac), (c_sp, c_sc, c_atac), (loader_sp, loader_sc, loader_atac)


def save_outputs(pred_sp, pred_atac, adata_sp, adata_sc, adata_atac, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out1 = output_dir / "{}_ATAC_to_ST.h5ad".format(prefix)
    out2 = output_dir / "{}_ST_to_ATAC.h5ad".format(prefix)

    ad.AnnData(
        pred_sp,
        obs=adata_atac.obs.copy(),
        var=adata_sp.var.copy(),
    ).write(out1)

    ad.AnnData(
        pred_atac,
        obs=adata_sp.obs.copy(),
        var=adata_atac.var.copy(),
        obsm=adata_sp.obsm.copy(),
    ).write(out2)

    return str(out1), str(out2)


def run_one_triplet(sp_path, sc_path, atac_path, output_dir, cfg):
    set_seed(cfg.seed, cfg.use_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    adata_sp, adata_sc, adata_atac = load_triplet(sp_path, sc_path, atac_path)

    if len(adata_sp) == 0 or len(adata_sc) == 0 or len(adata_atac) == 0:
        raise ValueError("One of the modalities is empty.")

    if not valid_subclass(
        adata_sc.obs["Subclass"],
        adata_atac.obs["Subclass"],
        adata_sp.obs["Subclass"],
    ):
        raise ValueError("Subclass mismatch across modalities.")

    adata_sp, adata_sc, adata_atac = preprocess_triplet(
        adata_sp,
        adata_sc,
        adata_atac,
        hvg_top_genes=cfg.hvg_top_genes,
    )

    (X_sp, X_sc, X_atac), (c_sp, c_sc, c_atac), (loader_sp, loader_sc, loader_atac) = make_loaders(
        adata_sp, adata_sc, adata_atac, batch_size=cfg.batch_size
    )

    vae_sp = VAE(X_sp.shape[1], int(torch.unique(c_sp).numel()), z=cfg.latent_dim).to(device)
    vae_sc = VAE(X_sc.shape[1], int(torch.unique(c_sc).numel()), z=cfg.latent_dim).to(device)
    vae_atac = VAE(X_atac.shape[1], int(torch.unique(c_atac).numel()), z=cfg.latent_dim).to(device)

    opt_sp = torch.optim.Adam(vae_sp.parameters(), lr=cfg.lr_vae)
    opt_sc = torch.optim.Adam(vae_sc.parameters(), lr=cfg.lr_vae)
    opt_atac = torch.optim.Adam(vae_atac.parameters(), lr=cfg.lr_vae)

    print("train ST")
    train_vae(vae_sp, loader_sp, opt_sp, device, epochs=cfg.vae_epochs, kl_weight=cfg.kl_weight)

    print("train RNA")
    train_vae(vae_sc, loader_sc, opt_sc, device, epochs=cfg.vae_epochs, kl_weight=cfg.kl_weight)

    print("train ATAC")
    train_vae(vae_atac, loader_atac, opt_atac, device, epochs=cfg.vae_epochs, kl_weight=cfg.kl_weight)

    with torch.no_grad():
        mu_sp = encode_mu(vae_sp, X_sp, c_sp, device)
        mu_sc = encode_mu(vae_sc, X_sc, c_sc, device)
        mu_atac = encode_mu(vae_atac, X_atac, c_atac, device)

    proj = Projector(z=cfg.latent_dim).to(device)
    opt_proj = torch.optim.Adam(proj.parameters(), lr=cfg.lr_proj)

    print("train projector")
    train_projector(
        proj,
        mu_sp,
        c_sp.to(device),
        mu_sc,
        c_sc.to(device),
        mu_atac,
        c_atac.to(device),
        opt_proj,
        device,
        epochs=cfg.proj_epochs,
    )

    with torch.no_grad():
        z_sp = proj(mu_sp)
        z_atac = proj(mu_atac)

        pred_sp = vae_sp.decode(z_atac, c_atac.to(device)).cpu().numpy()
        pred_atac = vae_atac.decode(z_sp, c_sp.to(device)).cpu().numpy()

    prefix = Path(sp_path).stem.replace("/", "_")
    out1, out2 = save_outputs(
        pred_sp,
        pred_atac,
        adata_sp,
        adata_sc,
        adata_atac,
        output_dir,
        prefix,
    )

    print("Saved:")
    print(out1)
    print(out2)
    return out1, out2