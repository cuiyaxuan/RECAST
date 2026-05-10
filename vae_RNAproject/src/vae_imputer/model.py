from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import graph_smoothness, weighted_smooth_l1, layer56_margin_loss


class STTopGeneImputer(nn.Module):
    def __init__(self, n_shared, n_target, n_subclass, n_layer=0, use_layer56=False, z_dim=64, dropout=0.1):
        super().__init__()
        self.n_shared = n_shared
        self.n_target = n_target
        self.n_subclass = n_subclass
        self.n_layer = n_layer
        self.use_layer56 = use_layer56
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_shared, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

        self.decoder_shared = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_shared),
            nn.Softplus(),
        )

        if n_target > 0:
            self.decoder_target = nn.Sequential(
                nn.Linear(z_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_target),
                nn.Softplus(),
            )
        else:
            self.decoder_target = None

        self.subclass_classifier = nn.Linear(z_dim, n_subclass) if n_subclass > 1 else None
        self.layer_classifier = nn.Linear(z_dim, n_layer) if n_layer > 1 else None
        self.layer56_classifier = nn.Linear(z_dim, 2) if use_layer56 else None

        nn.init.zeros_(self.fc_mu.bias)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, -8.0, 4.0)
        return mu, logvar

    def reparameterize(self, mu, logvar, sample=True):
        if self.training and sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode_shared(self, z):
        return self.decoder_shared(z)

    def decode_target(self, z):
        if self.decoder_target is None:
            return None
        return self.decoder_target(z)

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample=sample)
        shared_hat = self.decode_shared(z)
        target_hat = self.decode_target(z) if self.decoder_target is not None else None
        subclass_logits = self.subclass_classifier(z) if self.subclass_classifier is not None else None
        layer_logits = self.layer_classifier(z) if self.layer_classifier is not None else None
        layer56_logits = self.layer56_classifier(z) if self.layer56_classifier is not None else None
        return shared_hat, target_hat, mu, logvar, subclass_logits, layer_logits, layer56_logits


def sc_loss_fn(shared_hat, shared_x, target_hat, target_x, mu, logvar, subclass_logits, subclass_target,
               target_weights, kl_weight=1e-3, cls_weight=0.05):
    recon_shared = F.smooth_l1_loss(shared_hat, shared_x)
    if target_hat is not None and target_x.shape[1] > 0:
        recon_target = weighted_smooth_l1(target_hat, target_x, target_weights)
    else:
        recon_target = torch.tensor(0.0, device=shared_x.device)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if subclass_logits is not None:
        cls_loss = F.cross_entropy(subclass_logits, subclass_target)
    else:
        cls_loss = torch.tensor(0.0, device=shared_x.device)
    loss = recon_shared + recon_target + kl_weight * kl_loss + cls_weight * cls_loss
    return loss, recon_shared.detach(), recon_target.detach(), kl_loss.detach(), cls_loss.detach()


def layer56_binary_ce(layer56_logits, layer56_target, class_weights=None):
    if layer56_logits is None or layer56_target is None:
        return torch.tensor(0.0, device=layer56_logits.device if layer56_logits is not None else "cpu")
    valid = layer56_target >= 0
    if valid.sum().item() == 0:
        return torch.tensor(0.0, device=layer56_logits.device)
    if class_weights is not None:
        return F.cross_entropy(layer56_logits[valid], layer56_target[valid], weight=class_weights)
    return F.cross_entropy(layer56_logits[valid], layer56_target[valid])


def st_loss_fn(
    shared_hat, shared_x, mu, logvar, edges,
    layer_logits=None, layer_target=None, layer_class_weights=None,
    layer56_logits=None, layer56_target=None, layer56_class_weights=None,
    graph_weight=0.005, kl_weight=1e-6,
    layer_weight=0.8, layer56_weight=4.0, layer56_margin_weight=1.0,
):
    recon_shared = F.smooth_l1_loss(shared_hat, shared_x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    graph_loss = graph_smoothness(mu, edges)
    if layer_logits is not None and layer_target is not None:
        layer_loss = F.cross_entropy(layer_logits, layer_target, weight=layer_class_weights)
    else:
        layer_loss = torch.tensor(0.0, device=shared_x.device)
    if layer56_logits is not None and layer56_target is not None:
        layer56_ce = layer56_binary_ce(layer56_logits, layer56_target, class_weights=layer56_class_weights)
        layer56_margin = layer56_margin_loss(mu, layer56_target, margin=0.1)
    else:
        layer56_ce = torch.tensor(0.0, device=shared_x.device)
        layer56_margin = torch.tensor(0.0, device=shared_x.device)
    loss = (
        recon_shared
        + kl_weight * kl_loss
        + graph_weight * graph_loss
        + layer_weight * layer_loss
        + layer56_weight * layer56_ce
        + layer56_margin_weight * layer56_margin
    )
    return loss, recon_shared.detach(), kl_loss.detach(), graph_loss.detach(), layer_loss.detach(), layer56_ce.detach(), layer56_margin.detach()
