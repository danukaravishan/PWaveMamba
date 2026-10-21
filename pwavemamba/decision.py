"""Inference helpers and the window-level decision (top-K confidence + validation-tuned thresholds)."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import TOPK


@torch.no_grad()
def predict(model, X, device, bs=512):
    """X: (N, 3, L) preprocessed windows -> magnitude (N,), intensity (N,), per-sample probs (N, L, 3)."""
    model.eval()
    mags, ints, probs = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device)
        m, it, d = model(xb)
        mags.append(m.cpu().numpy())
        ints.append(it.cpu().numpy())
        probs.append(F.softmax(d, dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(mags), np.concatenate(ints), np.concatenate(probs)


def topk_conf(probs, k=TOPK):
    """Window confidence for P and S: mean of the K largest per-sample probabilities. probs: (N, L, 3)."""
    p = np.sort(probs[:, :, 1], axis=1)[:, -k:].mean(1)
    s = np.sort(probs[:, :, 2], axis=1)[:, -k:].mean(1)
    return p, s


def classify(cp, cs, tp, ts):
    """Window class (0 noise, 1 P, 2 S). If both confidences pass their thresholds, the larger margin wins."""
    hp, hs = cp >= tp, cs >= ts
    cls = np.zeros(len(cp), dtype=int)
    cls[hp & ~hs] = 1
    cls[hs & ~hp] = 2
    both = hp & hs
    cls[both] = np.where((cp - tp)[both] >= (cs - ts)[both], 1, 2)
    return cls
