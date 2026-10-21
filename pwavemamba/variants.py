"""Model variants of the ablation study (Table III) and the model factory used by every script.

    base         PWaveMamba (4 Mamba layers, multi-task, P-conditioned intensity pooling)
    no_pcond     intensity head pools with a plain mean instead of the P-probability weights
    det_only / int_only / mag_only   single-task training (same network, other losses switched off)
    lstm         each Mamba layer replaced by a causal LSTM of the same parameter budget
    transformer  each Mamba layer replaced by a causal Transformer block of the same budget
    mamba_l2     2 Mamba layers instead of 4
"""
import math

import torch
import torch.nn as nn

from .config import MODEL_HPARAMS, window_samples, DEFAULT_WINDOW
from .model import PWaveMamba

VARIANTS = ("base", "no_pcond", "det_only", "int_only", "mag_only", "lstm", "transformer", "mamba_l2")


class LSTMBlock(nn.Module):
    """Drop-in replacement for one Mamba layer: unidirectional (causal) LSTM + projection."""

    def __init__(self, d_model, hidden=68):
        super().__init__()
        self.lstm = nn.LSTM(d_model, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.proj(y)


class CausalTransformerBlock(nn.Module):
    """Drop-in replacement for one Mamba layer: causal multi-head attention + FFN.
    The first block adds sinusoidal positional encodings."""

    def __init__(self, d_model, n_heads=4, ff=236, max_len=4000, add_pe=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Linear(ff, d_model))
        self.add_pe = add_pe
        if add_pe:
            pos = torch.arange(max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        L = x.size(1)
        if self.add_pe:
            x = x + self.pe[:L].unsqueeze(0)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        h, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return h + self.ffn(self.norm(h))


class PWaveMambaVariant(PWaveMamba):
    """PWaveMamba with a switch that removes the P-probability conditioning of the intensity pooling."""

    def __init__(self, *a, p_conditioning=True, **kw):
        super().__init__(*a, **kw)
        self.p_conditioning = p_conditioning

    def forward(self, x):
        if self.p_conditioning:
            return super().forward(x)
        x = self.backbone(x)
        x_max, _ = x.max(dim=1)
        x_mean = x.mean(dim=1)
        ts_input = torch.cat([x, x_max.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        detection_ts = self.detection_head(ts_input).clamp(-50, 50)
        pooled = torch.cat([x_max, x_mean], -1)
        mag = self.magnitude_scalar_head(pooled).squeeze(-1)
        inten = self.intensity_scalar_head(pooled).squeeze(-1)  # global mean, no P weighting
        return mag, inten, detection_ts


def build_model(variant="base", window_id=DEFAULT_WINDOW, backend="auto"):
    """Build an (untrained) model of one variant for one window length."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    kw = dict(MODEL_HPARAMS)
    if variant == "mamba_l2":
        kw["n_layers"] = 2
    model = PWaveMambaVariant(window_size=window_samples(window_id), p_conditioning=(variant != "no_pcond"),
                              backend=backend, **kw)
    d, n = MODEL_HPARAMS["d_model"], MODEL_HPARAMS["n_layers"]
    if variant == "lstm":
        model.mamba_layers = nn.ModuleList([LSTMBlock(d) for _ in range(n)])
    elif variant == "transformer":
        model.mamba_layers = nn.ModuleList([CausalTransformerBlock(d, add_pe=(i == 0)) for i in range(n)])
    return model


def loss_weights(variant):
    """(lambda_det, lambda_int, lambda_mag). Single-task variants switch off the other losses."""
    return {"det_only": (1.0, 0.0, 0.0), "int_only": (0.0, 1.0, 0.0),
            "mag_only": (0.0, 0.0, 1.0)}.get(variant, (1.0, 1.0, 1.0))
