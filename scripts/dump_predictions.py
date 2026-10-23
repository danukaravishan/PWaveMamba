"""Per-window predictions of one checkpoint on any split (used to fix the alert level on the validation set).

    python scripts/dump_predictions.py --ckpt checkpoints/base_s42.pt --split validation

The window thresholds T_P/T_S are read from the evaluate.py output of the same checkpoint
(<out>/eval/<tag>.json, or --eval-json). Output: <out>/eval/<tag>_<split>_predictions.csv.gz
"""
import csv
import gzip
import json
import os

import _common
import torch

from pwavemamba.config import window_samples
from pwavemamba.data import load_split
from pwavemamba.decision import classify, predict, topk_conf
from pwavemamba.utils import load_checkpoint
from pwavemamba.variants import VARIANTS, build_model

ap = _common.base_parser(__doc__.split("\n")[0])
ap.add_argument("--ckpt", required=True)
ap.add_argument("--variant", choices=VARIANTS, help="default: read from the checkpoint")
ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
ap.add_argument("--eval-json", help="default: <out>/eval/<tag>.json")
ap.add_argument("--backend", default="auto", choices=["auto", "mamba_ssm", "reference"])
a = ap.parse_args()
tag = os.path.splitext(os.path.basename(a.ckpt))[0]
thr = tuple(json.load(open(a.eval_json or os.path.join(a.out, "eval", tag + ".json")))["test"]["thresholds_P_S"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
variant = a.variant or (ck.get("variant") if isinstance(ck, dict) else None) or "base"
model = build_model(variant, a.window, backend=a.backend).to(dev)
load_checkpoint(model, a.ckpt, dev)
X, ym, yi, yd, yt, ids = load_split(a.split, _common.segment_path(a), a.split_csv, window_samples(a.window))
mag, inten, probs = predict(model, X, dev)
cp, cs = topk_conf(probs)
cls = classify(cp, cs, *thr)
out = os.path.join(a.out, "eval", f"{tag}_{a.split}_predictions.csv.gz")
os.makedirs(os.path.dirname(out), exist_ok=True)
with gzip.open(out, "wt", newline="") as f:
    w = csv.writer(f)
    w.writerow(["window_id", "event_type", "pred_class", "conf_P", "conf_S", "mag_true", "mag_pred", "int_true_ln", "int_pred_ln"])
    for j, wid in enumerate(ids):
        w.writerow([wid, int(yt[j]), int(cls[j]), f"{cp[j]:.4f}", f"{cs[j]:.4f}", f"{ym[j]:.4f}", f"{mag[j]:.4f}",
                    f"{yi[j]:.4f}", f"{inten[j]:.4f}"])
print(f"[{tag}] {a.split}: {len(ids)} windows -> {out}")
