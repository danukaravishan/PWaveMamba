"""Evaluate a checkpoint: window thresholds tuned on validation (macro-F1), metrics on the test set.

    python scripts/evaluate.py --ckpt checkpoints/base_s42.pt
    python scripts/evaluate.py --ckpt checkpoints/base_s42_ID10.pt          # window inferred from the name

Outputs: <out>/eval/<tag>.json (metrics) and <out>/eval/<tag>_test_predictions.csv (one row per window).
Regression metrics use P windows only. Intensity is in ln(PGA [cm/s^2]) + 5 units.
"""
import csv
import math
import os
import re

import _common
import numpy as np
import torch

from pwavemamba.config import PGA_MMI5_CMS2, TOPK, window_samples
from pwavemamba.data import load_split, trace_of
from pwavemamba.decision import classify, predict, topk_conf
from pwavemamba.utils import env_info, load_checkpoint, save_json
from pwavemamba.variants import VARIANTS, build_model

LN_MMI5 = math.log(PGA_MMI5_CMS2) + 5.0  # 8.745


def reg_metrics(t, p):
    if len(t) == 0:
        return {}
    e = p - t
    ss = np.sum((t - t.mean()) ** 2)
    return {"n": int(len(t)), "r2": float(1 - np.sum(e ** 2) / ss) if ss > 0 else float("nan"),
            "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
            "std": float(np.std(e))}


def cls_metrics(y, yhat, k=3):
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y, yhat):
        cm[a, b] += 1
    per = {}
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                  "support": int(cm[c].sum())}
    rown = (cm / np.maximum(cm.sum(1, keepdims=True), 1)).round(4).tolist()
    return {"confusion": cm.tolist(), "confusion_row_norm": rown, "accuracy": float(np.trace(cm) / cm.sum()),
            "macro_f1": float(np.mean([per[c]["f1"] for c in range(k)])), "per_class": per}


def binary(y_true, y_pred):
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": pr, "recall": rc,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def tune(cp, cs, y):
    """Grid search of (T_P, T_S) in [0.05, 0.95], step 0.025, maximizing the validation macro-F1."""
    grid = np.round(np.arange(0.05, 0.951, 0.025), 3)
    best = (-1, None)
    for tp in grid:
        for ts in grid:
            f = cls_metrics(y, classify(cp, cs, tp, ts))["macro_f1"]
            if f > best[0]:
                best = (f, (float(tp), float(ts)))
    return best[1], best[0]


def pr_curve(conf, positive):
    out = []
    for t in np.round(np.arange(0.02, 1.0, 0.02), 2):
        pred = conf >= t
        tp = np.sum(pred & positive)
        fp = np.sum(pred & ~positive)
        fn = np.sum(~pred & positive)
        out.append([float(t), float(tp / (tp + fp)) if tp + fp else 1.0, float(tp / (tp + fn)) if tp + fn else 0.0])
    return out


def full_report(mag, inten, cp, cs, ytype, ymag, yint, thr, seed=0):
    cls = classify(cp, cs, *thr)
    rep = {"thresholds_P_S": thr, "K": TOPK, "detection_all": cls_metrics(ytype, cls)}
    rng = np.random.default_rng(seed)  # balanced subset: noise windows subsampled to the number of P windows
    idx_n, idx_p = np.where(ytype == 0)[0], np.where(ytype == 1)[0]
    keep = np.concatenate([rng.choice(idx_n, size=min(len(idx_n), len(idx_p)), replace=False), np.where(ytype != 0)[0]])
    rep["detection_balanced"] = cls_metrics(ytype[keep], cls[keep])
    rep["event_vs_noise"] = binary(ytype != 0, cls != 0)
    rep["pr_curve_P"] = pr_curve(cp, ytype == 1)
    rep["pr_curve_S"] = pr_curve(cs, ytype == 2)
    p = ytype == 1
    rep["intensity_ln"] = reg_metrics(yint[p], inten[p])
    rep["magnitude"] = reg_metrics(ymag[p], mag[p])
    bins = [(3, 4), (4, 5), (5, 6), (6, 9)]
    rep["magnitude_by_bin"] = {f"{lo}-{hi}": reg_metrics(ymag[p & (ymag >= lo) & (ymag < hi)],
                                                         mag[p & (ymag >= lo) & (ymag < hi)]) for lo, hi in bins}
    rep["intensity_by_mag_bin"] = {f"{lo}-{hi}": reg_metrics(yint[p & (ymag >= lo) & (ymag < hi)],
                                                             inten[p & (ymag >= lo) & (ymag < hi)]) for lo, hi in bins}
    rep["mmi5_exceedance_P_windows"] = binary(yint[p] > LN_MMI5, inten[p] > LN_MMI5)
    return rep, cls


def load_meta(path):
    meta = {}
    if os.path.isfile(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                meta[r["trace_name"]] = (r.get("epicentral_distance", ""), r.get("station", ""),
                                         r.get("magnitude_type", ""))
    return meta


def main():
    ap = _common.base_parser(__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", choices=VARIANTS, help="default: read from the checkpoint")
    ap.add_argument("--backend", default="auto", choices=["auto", "mamba_ssm", "reference"])
    a = ap.parse_args()
    tag = os.path.splitext(os.path.basename(a.ckpt))[0]
    m = re.search(r"_(ID\d+)$", tag)
    if m and a.window == "ID6":
        a.window = m.group(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    variant = a.variant or (ck.get("variant") if isinstance(ck, dict) else None) or "base"
    model = build_model(variant, a.window, backend=a.backend).to(dev)
    load_checkpoint(model, a.ckpt, dev)
    res = {"ckpt": os.path.basename(a.ckpt), "variant": variant, "window": a.window,
           "params": sum(p.numel() for p in model.parameters()), "env": env_info()}
    seg, L = _common.segment_path(a), window_samples(a.window)

    Xv, mv, iv, dv, tv, _ = load_split("validation", seg, a.split_csv, L)
    _, _, pv = predict(model, Xv, dev)
    cpv, csv_ = topk_conf(pv)
    thr, f1v = tune(cpv, csv_, tv)
    res["validation_macro_f1_at_tuned"] = f1v
    Xt, mt, it, dt, tt, ids = load_split("test", seg, a.split_csv, L)
    mag, inten, pt = predict(model, Xt, dev)
    cpt, cst = topk_conf(pt)
    res["test"], cls = full_report(mag, inten, cpt, cst, tt, mt, it, thr)
    save_json(res, os.path.join(a.out, "eval", f"{tag}.json"))

    meta = load_meta(os.path.join(a.data_dir, "metadata.csv"))
    with open(os.path.join(a.out, "eval", f"{tag}_test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "event_type", "pred_class", "conf_P", "conf_S", "mag_true", "mag_pred",
                    "int_true_ln", "int_pred_ln", "epicentral_distance_km", "station", "magnitude_type"])
        for j, wid in enumerate(ids):
            d, st, mtyp = meta.get(trace_of(wid), ("", "", ""))
            w.writerow([wid, int(tt[j]), int(cls[j]), f"{cpt[j]:.4f}", f"{cst[j]:.4f}", f"{mt[j]:.4f}", f"{mag[j]:.4f}",
                        f"{it[j]:.4f}", f"{inten[j]:.4f}", d, st, mtyp])
    d = res["test"]
    print(f"[{tag}] thr={thr} acc={d['detection_all']['accuracy']:.4f} intR2={d['intensity_ln'].get('r2', float('nan')):.4f} "
          f"magR2={d['magnitude'].get('r2', float('nan')):.4f} magMAE={d['magnitude'].get('mae', float('nan')):.4f}")


if __name__ == "__main__":
    main()
