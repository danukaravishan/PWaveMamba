"""2011 Mw 6.2 Christchurch case study (Section VII of the paper).

    python scripts/christchurch_eval.py --ckpt checkpoints/base_s42.pt --eval-json results/eval/base_s42.json

* Each record is scanned with 4 s windows (hop 0.1 s); every window is classified and its intensity and
  magnitude predicted. All per-window outputs are saved, so other causal decision rules can be applied
  offline (paper/christchurch_rules.py computes the first-exceedance rule of the paper).
* MMI 5 threshold in model space: ln(42.3 cm/s^2) + 5 = 8.745.
* Reference rule Pd (Wu & Kanamori 2005): vertical acceleration, pre-P mean removed, integrated twice with
  a causal 0.075 Hz 2nd-order Butterworth high-pass after each integration; Pd = max|d| in
  [tP, tP + tau], tau = 2 s and 3 s; alert if Pd > 0.5 cm.
* Hold rule (also reported): first P-classified window + hold H = 0, 0.5, 1 s, using the maximum intensity
  of the P windows that end by then. Undetected stations are kept (no alert).
* Per-station latency, lead times to S and to the time of peak horizontal acceleration, bootstrap 95% CIs
  for F1 and exact McNemar tests.
Inputs : data/christchurch/christchurch.hdf5 (group "data", one 3 x N dataset per record, attrs
         sampling_rate, channels) and data/christchurch/metadata.csv (trace_name, station,
         epicentral_distance, sampling_rate, trace_starttime, p/s arrival samples or times, pga_rotd50 [m/s^2]).
Outputs: <out>/christchurch/<tag>_stations.csv, <tag>_summary.json, <tag>_windows.csv.gz
"""
import argparse
import csv
import gzip
import json
import math
import os

import _common
import h5py
import numpy as np
import pandas as pd
import scipy.signal as sps
import torch
from scipy.stats import binom

from pwavemamba.decision import classify, predict, topk_conf
from pwavemamba.preprocessing import segment_preprocess_data
from pwavemamba.utils import load_checkpoint, save_json
from pwavemamba.variants import VARIANTS, build_model


def compute_rotd50_pga(a1, a2, angle_step=1):
    """RotD50 PGA: median over rotation angles of the peak of the rotated horizontal acceleration."""
    if a1.size == 0 or a2.size == 0:
        return float("nan")
    vals = []
    for ang in range(0, 180, angle_step):
        t = np.deg2rad(ang)
        vals.append(np.max(np.abs(a1 * np.cos(t) + a2 * np.sin(t))))
    return float(np.nanmedian(vals)) if vals else float("nan")

THR_LN = math.log(42.3) + 5.0
PGA_THR_MS2 = 0.423
PD_THR_M = 0.005
SR = 100.0
WIN = 400

ap = argparse.ArgumentParser(description="2011 Christchurch case study", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--variant", choices=VARIANTS, help="default: read from the checkpoint")
ap.add_argument("--eval-json", required=True, help="evaluate.py output of the same checkpoint; its tuned thresholds are used")
ap.add_argument("--stride", type=float, default=0.1, help="hop between successive 4 s windows (s)")
ap.add_argument("--hdf5", default=os.path.join(_common.ROOT, "data", "christchurch", "christchurch.hdf5"))
ap.add_argument("--meta", default=os.path.join(_common.ROOT, "data", "christchurch", "metadata.csv"))
ap.add_argument("--out", default=os.path.join(_common.ROOT, "outputs"))
ap.add_argument("--backend", default="auto", choices=["auto", "mamba_ssm", "reference"])
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt, tag = a.ckpt, os.path.splitext(os.path.basename(a.ckpt))[0]
thr = tuple(json.load(open(a.eval_json))["test"]["thresholds_P_S"])
_ck = torch.load(ckpt, map_location="cpu", weights_only=False)
variant = a.variant or (_ck.get("variant") if isinstance(_ck, dict) else None) or "base"
model = build_model(variant, "ID6", backend=a.backend).to(dev).eval()
load_checkpoint(model, ckpt, dev)

meta = pd.read_csv(a.meta).set_index("trace_name")


def arrival(row, tcol, scol):
    sr = row.get("sampling_rate", np.nan)
    for col in (scol,):
        v = row.get(col, np.nan)
        if pd.notna(v) and pd.notna(sr) and float(sr) > 0:
            return float(v) / float(sr)
    t, t0 = row.get(tcol, None), row.get("trace_starttime", None)
    if isinstance(t, str) and isinstance(t0, str) and t and t0:
        return (pd.Timestamp(t) - pd.Timestamp(t0)).total_seconds()
    return np.nan


def hp(x, fc=0.075, fs=SR):
    b, a_ = sps.butter(2, fc / (fs / 2), btype="highpass")
    return sps.lfilter(b, a_, x)


def pd_value(z, ip, tau):
    if ip <= 0 or ip + int(tau * SR) > len(z):
        return np.nan
    acc = z - np.mean(z[max(0, ip - int(5 * SR)):ip])
    vel = hp(np.cumsum(acc) / SR)
    dis = hp(np.cumsum(vel) / SR)
    return float(np.max(np.abs(dis[ip:ip + int(tau * SR)])))


rows, win_rows = [], []
with h5py.File(a.hdf5, "r") as h:
    for tr in h["data"].keys():
        if tr not in meta.index:
            continue
        row = meta.loc[tr]
        ds = h["data"][tr]
        w = np.asarray(ds, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != 3:
            continue
        sr0 = float(ds.attrs.get("sampling_rate", row.get("sampling_rate", SR)))
        if sr0 != SR:
            w = sps.resample(w, int(w.shape[1] * SR / sr0), axis=1)
        chans = [c.strip().upper() for c in str(ds.attrs.get("channels", "")).split(",") if c.strip()]
        iz = next((i for i, c in enumerate(chans) if c.endswith("Z")), 2)
        ih = [i for i in range(3) if i != iz]
        tP = arrival(row, "phasenet_p_arrival_time", "phasenet_p_arrival_sample")
        if not np.isfinite(tP):
            tP = arrival(row, "p_arrival_time", "p_arrival_sample")
        tS = arrival(row, "phasenet_s_arrival_time", "phasenet_s_arrival_sample")
        if not np.isfinite(tS):
            tS = arrival(row, "s_arrival_time", "s_arrival_sample")
        pga_meta = float(row.get("pga_rotd50", np.nan))
        ip = int(round(tP * SR)) if np.isfinite(tP) else -1
        pga_full = compute_rotd50_pga(w[ih[0], ip:], w[ih[1], ip:]) if ip >= 0 else np.nan
        hv = np.hypot(w[ih[0]], w[ih[1]])
        t_pga = float(np.argmax(hv[max(ip, 0):]) + max(ip, 0)) / SR
        # sliding windows
        step = max(1, int(round(a.stride * SR)))
        starts = np.arange(0, w.shape[1] - WIN + 1, step)
        if len(starts) == 0:
            continue
        X = np.stack([segment_preprocess_data(w[:, s:s + WIN].astype(np.float32)) for s in starts]).astype(np.float32)
        mag, inten, probs = predict(model, X, dev)
        cp, cs = topk_conf(probs)
        cls = classify(cp, cs, *thr)
        ends = (starts + WIN) / SR
        for j in range(len(starts)):  # per-window outputs, for offline analysis of other decision rules
            win_rows.append((tr, f"{ends[j]:.2f}", int(cls[j]), f"{cp[j]:.4f}", f"{cs[j]:.4f}",
                             f"{inten[j]:.4f}", f"{mag[j]:.4f}"))
        r = {"trace_name": tr, "station": row.get("station", ""), "channels": ",".join(chans),
             "epicentral_distance_km": row.get("epicentral_distance", np.nan),
             "tP": tP, "tS": tS, "t_PGA": t_pga, "pga_meta_cms2": 100 * pga_meta if np.isfinite(pga_meta) else np.nan,
             "pga_fromP_cms2": 100 * pga_full if np.isfinite(pga_full) else np.nan}
        for col in meta.columns:
            if "lat" in col.lower() or "lon" in col.lower():
                r[col] = row.get(col)
        r["damaging"] = bool(np.isfinite(pga_meta) and pga_meta > PGA_THR_MS2)
        pw = np.where(cls == 1)[0]
        r["detected"] = len(pw) > 0
        if len(pw):
            t_first = ends[pw[0]]
            r["t_first_P_window_end"] = t_first
            r["early_trigger"] = bool(np.isfinite(tP) and t_first < tP)
            for H in (0.0, 0.5, 1.0):
                sel = pw[ends[pw] <= t_first + H + 1e-9]
                k = f"H{H:g}"
                r[f"int_max_{k}"] = float(inten[sel].max())
                r[f"int_mean_{k}"] = float(inten[sel].mean())
                r[f"mag_max_{k}"] = float(mag[sel].max())
                r[f"mag_mean_{k}"] = float(mag[sel].mean())
                r[f"t_dec_{k}"] = t_first + H
                r[f"alert_{k}"] = bool(r[f"int_max_{k}"] > THR_LN)
                r[f"latency_{k}"] = t_first + H - tP if np.isfinite(tP) else np.nan
                r[f"lead_S_{k}"] = tS - (t_first + H) if np.isfinite(tS) else np.nan
                r[f"lead_PGA_{k}"] = t_pga - (t_first + H)
            runs = np.split(pw, np.where(np.diff(pw) != 1)[0] + 1)
            best = max(runs, key=len)
            r["int_longest_run_rule"] = float(inten[best].max())
            r["alert_longest_run_rule"] = bool(r["int_longest_run_rule"] > THR_LN)
            r["t_dec_longest_run_rule"] = float(ends[best[-1]])
        else:
            for H in (0.0, 0.5, 1.0):
                r[f"alert_H{H:g}"] = False
            r["alert_longest_run_rule"] = False
        z = w[iz]
        for tau in (2, 3):
            v = pd_value(z, ip, tau)
            r[f"Pd{tau}_cm"] = 100 * v if np.isfinite(v) else np.nan
            r[f"alert_Pd{tau}"] = bool(np.isfinite(v) and v > PD_THR_M)
        rows.append(r)

df = pd.DataFrame(rows)
odir = os.path.join(a.out, "christchurch")
os.makedirs(odir, exist_ok=True)
df.to_csv(os.path.join(odir, f"{tag}_stations.csv"), index=False)
with gzip.open(os.path.join(odir, f"{tag}_windows.csv.gz"), "wt", newline="") as fz:
    w = csv.writer(fz)
    w.writerow(["trace_name", "window_end_s", "class", "conf_P", "conf_S", "int_pred_ln", "mag_pred"])
    w.writerows(win_rows)


def conf(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn, tn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def boot_f1(y, p, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    f = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        f.append(conf(y[i], p[i])["f1"])
    return [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]


def mcnemar(y, p1, p2):
    c1, c2 = (np.asarray(p1) == y), (np.asarray(p2) == y)
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    pval = min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5)) if b + c else 1.0
    return {"model_right_pd_wrong": b, "model_wrong_pd_right": c, "p_value_exact": float(pval)}


summary = {"ckpt": ckpt, "thresholds_P_S": thr, "stride_s": a.stride, "threshold_ln": THR_LN,
           "n_traces": int(len(df)), "n_damaging": int(df["damaging"].sum()), "n_detected": int(df["detected"].sum())}
valid = df[np.isfinite(df["pga_meta_cms2"]) & np.isfinite(df["tP"])]
for name, sub in (("all", valid), ("detected", valid[valid["detected"]])):
    y = sub["damaging"].values
    s = {}
    for col in ["alert_H0", "alert_H0.5", "alert_H1", "alert_longest_run_rule", "alert_Pd2", "alert_Pd3"]:
        if col in sub:
            p = sub[col].fillna(False).values.astype(bool)
            s[col] = conf(y, p) | {"f1_ci95": boot_f1(y, p)}
    s["mcnemar_H1_vs_Pd3"] = mcnemar(y, sub["alert_H1"].fillna(False).values.astype(bool), sub["alert_Pd3"].values)
    s["mcnemar_H1_vs_Pd2"] = mcnemar(y, sub["alert_H1"].fillna(False).values.astype(bool), sub["alert_Pd2"].values)
    s["n"] = int(len(sub))
    summary[name] = s
det = valid[valid["detected"]]
for H in ("H0", "H0.5", "H1"):
    if f"latency_{H}" in det:
        lat = det[f"latency_{H}"].dropna()
        tpm = det[det["damaging"] & det[f"alert_{H}"].fillna(False).astype(bool)]
        summary[f"latency_{H}"] = {"median_s": float(lat.median()), "q25": float(lat.quantile(.25)), "q75": float(lat.quantile(.75)),
                                   "lead_PGA_TP_median_s": float(tpm[f"lead_PGA_{H}"].median()) if len(tpm) else None,
                                   "lead_PGA_TP_positive_frac": float((tpm[f"lead_PGA_{H}"] > 0).mean()) if len(tpm) else None,
                                   "lead_S_TP_median_s": float(tpm[f"lead_S_{H}"].median()) if len(tpm) else None}
    if f"mag_max_{H}" in det:
        summary[f"magnitude_{H}"] = {"median_max": float(det[f"mag_max_{H}"].median()), "median_mean": float(det[f"mag_mean_{H}"].median())}
summary["early_triggers"] = int(det.get("early_trigger", pd.Series(dtype=bool)).fillna(False).sum())
save_json(summary, os.path.join(odir, f"{tag}_summary.json"))
print(json.dumps(summary, indent=1, default=float)[:3000])
