"""Train one model variant on the event-disjoint split with the paper's recipe.

    python scripts/train.py --variant base --seed 42
    python scripts/train.py --variant base --seed 42 --window ID10      # window-length study (2 s + 3 s)

Recipe: Adam (lr 1.0406e-3, weight decay 1.288e-5), one-cycle schedule (pct_start 0.1, div 10,
final_div 100, cosine), gradient-norm clip 1.0, batch 64, up to 300 epochs, early stopping with
patience 200 on the validation loss; the best-validation weights are kept. Equal class weights in the
cross-entropy and equal task weights. Needs a GPU with `mamba_ssm` for the Mamba variants.

Outputs: <out>/models/<variant>_s<seed>[_<window>].pt and <out>/train/<same>.json (loss history).
"""
import os
import time

import _common
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pwavemamba.config import TRAIN, tag_suffix, window_samples
from pwavemamba.data import load_split
from pwavemamba.losses import total_loss
from pwavemamba.utils import env_info, save_json, set_seed
from pwavemamba.variants import VARIANTS, build_model, loss_weights

ap = _common.base_parser(__doc__.split("\n")[0])
ap.add_argument("--variant", required=True, choices=VARIANTS)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
ap.add_argument("--patience", type=int, default=TRAIN["patience"])
ap.add_argument("--batch", type=int, default=TRAIN["batch_size"])
ap.add_argument("--backend", default="auto", choices=["auto", "mamba_ssm", "reference"])
a = ap.parse_args()

tag = f"{a.variant}_s{a.seed}{tag_suffix(a.window)}"
mdir = os.path.join(a.out, "models")
os.makedirs(mdir, exist_ok=True)
set_seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg, L = _common.segment_path(a), window_samples(a.window)

Xtr, mtr, itr, dtr, _, _ = load_split("train", seg, a.split_csv, L)
Xva, mva, iva, dva, _, _ = load_split("validation", seg, a.split_csv, L)
mk = lambda X, m, i, d, sh: DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(m), torch.from_numpy(i),
                                                     torch.from_numpy(d)), batch_size=a.batch, shuffle=sh)
tl, vl = mk(Xtr, mtr, itr, dtr, True), mk(Xva, mva, iva, dva, False)

model = build_model(a.variant, a.window, backend=a.backend).to(dev)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[{tag}] params={n_params:,} train={len(Xtr)} val={len(Xva)} device={dev}")
lr = TRAIN["learning_rate"]
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TRAIN["weight_decay"])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=a.epochs, steps_per_epoch=len(tl), pct_start=0.1,
                                            div_factor=10.0, final_div_factor=100.0, anneal_strategy="cos")
ce = nn.CrossEntropyLoss(weight=torch.ones(3, device=dev))
lam = loss_weights(a.variant)

hist = {"train": [], "val": [], "val_parts": []}
best, best_ep, best_sd, wait = float("inf"), -1, None, 0
t0 = time.time()
for ep in range(a.epochs):
    model.train()
    s = 0.0
    for xb, mb, ib, db in tl:
        xb, mb, ib, db = xb.to(dev), mb.to(dev), ib.to(dev), db.to(dev).long()
        opt.zero_grad()
        loss, _ = total_loss(model(xb), mb, ib, db, ce, lam, delta=TRAIN["huber_delta"])
        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss at epoch {ep + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
        opt.step()
        sched.step()
        s += loss.item()
    model.eval()
    v, parts = 0.0, np.zeros(3)
    with torch.no_grad():
        for xb, mb, ib, db in vl:
            xb, mb, ib, db = xb.to(dev), mb.to(dev), ib.to(dev), db.to(dev).long()
            loss, p = total_loss(model(xb), mb, ib, db, ce, lam, delta=TRAIN["huber_delta"])
            v += loss.item()
            parts += p
    v /= max(1, len(vl))
    hist["train"].append(s / max(1, len(tl)))
    hist["val"].append(v)
    hist["val_parts"].append((parts / max(1, len(vl))).tolist())
    if v < best - 1e-8:
        best, best_ep, wait = v, ep + 1, 0
        best_sd = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    else:
        wait += 1
    print(f"[{tag}] ep {ep + 1} train {hist['train'][-1]:.5f} val {v:.5f} best@{best_ep}", flush=True)
    if wait >= a.patience:
        print(f"[{tag}] early stop at epoch {ep + 1}")
        break

ck = os.path.join(mdir, f"{tag}.pt")
torch.save({"model_state_dict": best_sd, "variant": a.variant, "seed": a.seed, "window": a.window,
            "best_epoch": best_ep, "epochs_run": len(hist["train"])}, ck)
save_json({"variant": a.variant, "seed": a.seed, "window": a.window, "params": n_params, "best_epoch": best_ep,
           "epochs_run": len(hist["train"]), "best_val_loss": best, "train_seconds": time.time() - t0,
           "lambdas_det_int_mag": lam, "env": env_info(), "history": hist},
          os.path.join(a.out, "train", f"{tag}.json"))
print(f"[{tag}] saved {ck} (best epoch {best_ep}, {time.time() - t0:.0f} s)")
