from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .model import sc_loss_fn, st_loss_fn
from .utils import freeze_module


@dataclass
class TrainResult:
    best_state: dict
    model: torch.nn.Module


def pretrain_sc(model, train_loader, val_loader, optimizer, device, target_weights, num_epochs=40, warmup_epochs=10,
                patience=8, grad_clip=5.0, cls_weight=0.05):
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(num_epochs):
        model.train()
        kl_weight = 1e-3 * (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1e-3
        total_train = total_val = 0.0
        n_batches = 0
        for x_shared, x_target, subclass in train_loader:
            x_shared = x_shared.to(device)
            x_target = x_target.to(device)
            subclass = subclass.to(device)
            drop_mask = (torch.rand_like(x_shared) > 0.1).float()
            x_shared_in = x_shared * drop_mask
            shared_hat, target_hat, mu, logvar, subclass_logits, _, _ = model(x_shared_in, sample=True)
            loss, *_ = sc_loss_fn(
                shared_hat, x_shared, target_hat, x_target, mu, logvar,
                subclass_logits, subclass, target_weights,
                kl_weight=kl_weight, cls_weight=cls_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_train += loss.item()
            n_batches += 1
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_batches = 0
            for x_shared, x_target, subclass in val_loader:
                x_shared = x_shared.to(device)
                x_target = x_target.to(device)
                subclass = subclass.to(device)
                shared_hat, target_hat, mu, logvar, subclass_logits, _, _ = model(x_shared, sample=False)
                loss, *_ = sc_loss_fn(
                    shared_hat, x_shared, target_hat, x_target, mu, logvar,
                    subclass_logits, subclass, target_weights,
                    kl_weight=kl_weight, cls_weight=cls_weight,
                )
                val_total += loss.item()
                val_batches += 1
        val_loss = val_total / max(1, val_batches)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"[sc pretrain] Epoch {epoch:02d} | train={total_train/max(1,n_batches):.4f} | val={val_loss:.4f}")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping (sc pretrain) at epoch {epoch:02d}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_state


def finetune_st(model, adata_sp, X_sp_shared, layer_codes_t, layer56_codes_t, layer_class_weights, layer56_class_weights,
                device, num_epochs=25, patience=6, lr=2e-4):
    from .utils import build_knn_edges
    if "spatial" not in adata_sp.obsm_keys():
        raise ValueError("adata_sp.obsm['spatial'] not found. Please provide spatial coordinates.")
    spatial_coords = np.asarray(adata_sp.obsm["spatial"])
    edges = build_knn_edges(spatial_coords, k=2)
    edges_t = torch.tensor(edges, dtype=torch.long, device=device)

    if model.decoder_target is not None:
        freeze_module(model.decoder_target, freeze=True)
    if model.subclass_classifier is not None:
        freeze_module(model.subclass_classifier, freeze=True)
    freeze_module(model.encoder, freeze=False)
    freeze_module(model.fc_mu, freeze=False)
    freeze_module(model.fc_logvar, freeze=False)
    freeze_module(model.decoder_shared, freeze=False)
    if model.layer_classifier is not None:
        freeze_module(model.layer_classifier, freeze=False)
    if model.layer56_classifier is not None:
        freeze_module(model.layer56_classifier, freeze=False)

    optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, weight_decay=1e-4)
    X_sp_shared_t = torch.tensor(X_sp_shared, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(num_epochs):
        model.train()
        mu, logvar = model.encode(X_sp_shared_t)
        shared_hat = model.decode_shared(mu)
        layer_logits = model.layer_classifier(mu) if model.layer_classifier is not None else None
        layer56_logits = model.layer56_classifier(mu) if model.layer56_classifier is not None else None
        loss, *parts = st_loss_fn(
            shared_hat, X_sp_shared_t, mu, logvar, edges_t,
            layer_logits=layer_logits,
            layer_target=layer_codes_t,
            layer_class_weights=layer_class_weights,
            layer56_logits=layer56_logits,
            layer56_target=layer56_codes_t,
            layer56_class_weights=layer56_class_weights,
            graph_weight=0.005,
            kl_weight=1e-6,
            layer_weight=0.8 if layer_codes_t is not None else 0.0,
            layer56_weight=4.0 if layer56_codes_t is not None else 0.0,
            layer56_margin_weight=1.0 if layer56_codes_t is not None else 0.0,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(filter(lambda p: p.requires_grad, model.parameters())), 5.0)
        optimizer.step()
        st_val = loss.item()
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            print(f"[ST finetune] Epoch {epoch:02d} | loss={loss.item():.4f}")
        if st_val < best_val - 1e-5:
            best_val = st_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping (ST finetune) at epoch {epoch:02d}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_state
