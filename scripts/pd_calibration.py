"""Pd on the NZ records: calibrate the Pd alert threshold for MMI 5 on the TRAINING split, evaluate on the
test split (batch 3). CPU only, standard scientific stack + obspy.

    python scripts/pd_calibration.py                   # all records with a P pick
    python scripts/pd_calibration.py --max-records 20000

The classical threshold Pd > 0.5 cm (Wu & Kanamori 2005) targets damaging shaking, a higher level than MMI 5
(42.3 cm/s^2); this script fits the Pd threshold for MMI 5 on the NZ training records (Section VI-E of the paper).

Pd: same definition as scripts/christchurch_eval.py (vertical acceleration resampled to 100 Hz, pre-P mean removed,
integrated twice with a causal 0.075 Hz 2nd-order Butterworth high-pass after each integration, max |d| in
[tP, tP + tau], tau = 1, 1.5, 2, 2.5, 3 s). The same Pd values are computed for the Christchurch records (tP from
<out>/christchurch/base_s42_stations.csv, written by christchurch_eval.py), so that Pd and PWaveMamba can be compared at every deadline with
thresholds fixed on NZ data. PGA: the training target (RotD50 from tP + min(S-P, 3 s), as in
scripts/extract_segments.py). P pick: metadata column new_p_arrival_sample (the pick used for training).
Outputs: <out>/pd/pd_records.csv.gz (one row per record), christchurch_pd.csv, pd_calibration.json.
"""
import argparse, csv, gzip, json, math, os
import h5py
import numpy as np
import scipy.signal as sps
from obspy.geodetics import kilometers2degrees
from obspy.taup import TauPyModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 100.0
THR_CMS2 = 42.3
TAUS = (1.0, 1.5, 2.0, 2.5, 3.0)
_TAUP = None

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
ap.add_argument("--split-csv", default=os.path.join(ROOT, "splits", "event_split.csv"))
ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
ap.add_argument("--max-records", type=int, default=0, help="0 = all")
a = ap.parse_args()
OUT, SPLIT_CSV = os.path.join(a.out, "pd"), a.split_csv
a.waveforms, a.metadata = os.path.join(a.data_dir, "waveforms.hdf5"), os.path.join(a.data_dir, "metadata.csv")
os.makedirs(OUT, exist_ok=True)


# ---------- PGA target, copied from src/extract_data.py ----------
def compute_rotd50_pga(a1, a2, angle_step=1):
    if a1.size == 0 or a2.size == 0:
        return float("nan")
    angles = np.deg2rad(np.arange(0, 180, angle_step, dtype=float))
    vals = [np.max(np.abs(a1 * np.cos(th) + a2 * np.sin(th))) for th in angles]
    return float(np.nanmedian(vals)) if vals else float("nan")


def estimate_s_arrival_offset(dist_km, depth_km):
    global _TAUP
    try:
        if _TAUP is None:
            _TAUP = TauPyModel(model="iasp91")
        arr = _TAUP.get_travel_times(source_depth_in_km=depth_km, distance_in_degree=kilometers2degrees(dist_km),
                                     phase_list=["p", "P", "s", "S"])
        tp = ts = None
        for x in arr:
            n = x.name.upper()
            if tp is None and n.startswith("P"):
                tp = x.time
            if ts is None and n.startswith("S"):
                ts = x.time
            if tp is not None and ts is not None:
                break
        return ts - tp if tp is not None and ts is not None else None
    except Exception:
        return None


def calculate_pga(data, channels, ip, sr, dist_km=None, depth_km=None, default_window_sec=3.0):
    if data.ndim != 2 or data.shape[0] < 2:
        return float("nan")
    off = default_window_sec
    if dist_km is not None and depth_km is not None:
        sp = estimate_s_arrival_offset(dist_km, depth_km)
        if sp is not None and 0 < sp < default_window_sec:
            off = sp
    start = ip + int(off * sr)
    if start >= data.shape[1]:
        return float("nan")
    h = [i for i, c in enumerate(channels) if not c.strip().upper().endswith("Z")]
    if len(h) < 2:
        return float("nan")
    return compute_rotd50_pga(data[h[0], start:], data[h[1], start:])


# ---------- Pd, copied from christchurch_eval.py ----------
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


fnum = lambda v: float(v) if v not in ("", None, "nan", "NaN") else float("nan")
split = {r["event_id"]: r["split"] for r in csv.DictReader(open(SPLIT_CSV))}
meta = {r["trace_name"]: r for r in csv.DictReader(open(a.metadata))}
rows, n = [], 0
with h5py.File(a.waveforms, "r") as f:
    g = f["data"]
    for tr, m in meta.items():
        if a.max_records and n >= a.max_records:
            break
        s = split.get(tr.split("_")[0])
        pick = fnum(m.get("new_p_arrival_sample", ""))
        if s is None or not np.isfinite(pick) or tr not in g:
            continue
        ds = g[tr]
        w = np.asarray(ds, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != 3:
            continue
        sr0 = float(ds.attrs["sampling_rate"])
        chans = [c.strip().upper() for c in str(ds.attrs.get("channels", "")).split(",") if c.strip()]
        ip0 = int(pick)
        if ip0 <= 0 or ip0 >= w.shape[1]:
            continue
        dist, dep = fnum(m.get("epicentral_distance", "")), fnum(m.get("event_depth", ""))
        pga = calculate_pga(w, chans, ip0, int(sr0), dist if np.isfinite(dist) else None, dep if np.isfinite(dep) else None)
        if not np.isfinite(pga):
            continue
        w100 = sps.resample(w, int(w.shape[1] * SR / sr0), axis=1) if sr0 != SR else w
        ip = int(ip0 * w100.shape[1] / w.shape[1])
        iz = next((i for i, c in enumerate(chans) if c.endswith("Z")), 2)
        rows.append((tr, s, fnum(m.get("magnitude", "")), dist, 100 * pga) + tuple(100 * pd_value(w100[iz], ip, t) for t in TAUS))
        n += 1
        if n % 5000 == 0:
            print(f"{n} records", flush=True)

with gzip.open(os.path.join(OUT, "pd_records.csv.gz"), "wt", newline="") as fz:
    wr = csv.writer(fz)
    wr.writerow(["trace_name", "split", "magnitude", "epicentral_distance_km", "pga_cms2"] + [f"Pd{t:g}_cm" for t in TAUS])
    wr.writerows(rows)


def conf(y, p):
    tp, fp, fn = int(np.sum(y & p)), int(np.sum(~y & p)), int(np.sum(y & ~p))
    return {"TP": tp, "FP": fp, "FN": fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def auc(y, s):
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


res = {"n_records": len(rows), "threshold_cms2": THR_CMS2, "by_tau": {}}
R = {s: [r for r in rows if r[1] == s] for s in ("train", "validation", "test")}
for k, tau in ((5 + i, f"{t:g}") for i, t in enumerate(TAUS)):
    tr = np.array([[r[4], r[k]] for r in R["train"] if np.isfinite(r[k]) and r[k] > 0])
    y = tr[:, 0] > THR_CMS2
    b, a0 = np.polyfit(np.log10(tr[:, 1]), np.log10(tr[:, 0]), 1)       # log10 PGA = a0 + b log10 Pd
    thr_reg = 10 ** ((math.log10(THR_CMS2) - a0) / b)
    grid = np.quantile(tr[:, 1], np.linspace(0.5, 0.9999, 2000))
    f1s = [conf(y, tr[:, 1] > g)["f1"] for g in grid]
    thr_f1 = float(grid[int(np.argmax(f1s))])
    out = {"train_n": int(len(tr)), "train_n_above": int(y.sum()), "regression_log10PGA_on_log10Pd": [float(a0), float(b)],
           "thr_regression_cm": float(thr_reg), "thr_trainF1_cm": thr_f1, "train_AUC": auc(y, tr[:, 1])}
    for s in ("validation", "test"):
        te = np.array([[r[4], r[k]] for r in R[s] if np.isfinite(r[k])])
        yt = te[:, 0] > THR_CMS2
        out[s] = {"n": int(len(te)), "n_above": int(yt.sum()), "AUC": auc(yt, te[:, 1]),
                  "at_0.5cm": conf(yt, te[:, 1] > 0.5), "at_regression_thr": conf(yt, te[:, 1] > thr_reg),
                  "at_trainF1_thr": conf(yt, te[:, 1] > thr_f1)}
    res["by_tau"][tau] = out

# ---------- Christchurch Pd at the same windows (tP and damaging flag from the station table) ----------
chc_h5, chc_st = os.path.join(a.data_dir, "christchurch", "christchurch.hdf5"), os.path.join(a.out, "christchurch", "base_s42_stations.csv")
if os.path.isfile(chc_h5) and os.path.isfile(chc_st):
    st = {r["trace_name"]: r for r in csv.DictReader(open(chc_st))}
    crow = []
    with h5py.File(chc_h5, "r") as h:
        for tr, r in st.items():
            tP, pga = fnum(r["tP"]), fnum(r["pga_meta_cms2"])
            if not (np.isfinite(tP) and np.isfinite(pga)) or tr not in h["data"]:
                continue
            ds = h["data"][tr]; w = np.asarray(ds, dtype=np.float64)
            sr0 = float(ds.attrs.get("sampling_rate", SR))
            if sr0 != SR:
                w = sps.resample(w, int(w.shape[1] * SR / sr0), axis=1)
            chans = [c.strip().upper() for c in str(ds.attrs.get("channels", "")).split(",") if c.strip()]
            iz = next((i for i, c in enumerate(chans) if c.endswith("Z")), 2)
            ip = int(round(tP * SR))
            crow.append([tr, pga, r["damaging"]] + [100 * pd_value(w[iz], ip, t) for t in TAUS])
    with open(os.path.join(OUT, "christchurch_pd.csv"), "w", newline="") as fc:
        wc = csv.writer(fc); wc.writerow(["trace_name", "pga_meta_cms2", "damaging"] + [f"Pd{t:g}_cm" for t in TAUS]); wc.writerows(crow)
    y = np.array([c[2] == "True" for c in crow])
    res["christchurch"] = {"n": len(crow), "n_damaging": int(y.sum())}
    for i, t in enumerate(TAUS):
        v = np.array([c[3 + i] if np.isfinite(c[3 + i]) else -np.inf for c in crow]); k = f"{t:g}"
        res["christchurch"][k] = {"AUC": auc(y, v), "at_0.5cm": conf(y, v > 0.5),
                                  "at_regression_thr": conf(y, v > res["by_tau"][k]["thr_regression_cm"]),
                                  "at_trainF1_thr": conf(y, v > res["by_tau"][k]["thr_trainF1_cm"])}
json.dump(res, open(os.path.join(OUT, "pd_calibration.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
