#!/usr/bin/env python3
"""Supplementary Material tables and figures from the saved run outputs.

    python3 paper/si_tables.py [results] [data]

Reads <results>/{eval,train,christchurch,optuna}, splits/event_split.csv, <data>/metadata.csv and
<data>/waveforms.hdf5 (one example record, Fig. S1; skipped if absent).
Writes <results>/paper/si_tables.json and <results>/paper/figures/figS*.pdf. Needs numpy, matplotlib, h5py.
"""
import csv, glob, gzip, json, math, os, sqlite3, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data")
SPLIT = os.path.join(ROOT, "splits", "event_split.csv")
FIG = os.path.join(OUT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
TAU = math.log(42.3) + 5.0
SEEDS = ["base_s42", "base_s7", "base_s123"]
C = dict(b="#0072B2", o="#E69F00", g="#009E73", v="#D55E00", p="#CC79A7", s="#56B4E9", k="#555555")
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
f = lambda x: float(x) if x not in ("", "nan", "None", "NaN", None) else float("nan")
T = {}

# ---------- helpers ----------
def classify(cp, cs, tp, ts):
    p, s = cp >= tp, cs >= ts
    if p and s:
        return 1 if cp - tp >= cs - ts else 2
    return 1 if p else (2 if s else 0)

def macro_f1(y, yh, k=3):
    f1 = []
    for c in range(k):
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0)
    return float(np.mean(f1))

def f1_bin(y, p):
    tp = sum(a and b for a, b in zip(y, p)); fp = sum((not a) and b for a, b in zip(y, p)); fn = sum(a and (not b) for a, b in zip(y, p))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0

def load_ev(tag):
    return json.load(open(os.path.join(OUT, "eval", tag + ".json")))

def preds(tag):
    p = os.path.join(OUT, "eval", tag + "_test_predictions.csv")
    return list(csv.DictReader(gzip.open(p + ".gz", "rt") if os.path.isfile(p + ".gz") else open(p)))

def chc(tag):
    st = {r["trace_name"]: r for r in csv.DictReader(open(os.path.join(OUT, "christchurch", tag + "_stations.csv")))}
    W = defaultdict(list)
    for r in csv.DictReader(gzip.open(os.path.join(OUT, "christchurch", tag + "_windows.csv.gz"), "rt")):
        W[r["trace_name"]].append((f(r["window_end_s"]), f(r["conf_P"]), f(r["conf_S"]), f(r["int_pred_ln"]), f(r["mag_pred"])))
    V = [t for t, r in st.items() if not math.isnan(f(r["pga_meta_cms2"])) and not math.isnan(f(r["tP"]))]
    return st, W, V

def fe_times(st, W, V, tp_, ts_):
    out = {}
    for t in V:
        tP = f(st[t]["tP"]); hit = None
        for e, cp, cs, i, m in sorted(W[t]):
            if classify(cp, cs, tp_, ts_) == 1 and i > TAU:
                hit = (e - tP, m, i); break
        out[t] = hit
    return out

# ---------- T1: regression/detection summary with RMSE (3 seeds) ----------
rows = defaultdict(list)
for tag in SEEDS:
    d = load_ev(tag)["test"]
    rows["acc"].append(d["detection_all"]["accuracy"]); rows["macro_f1"].append(d["detection_all"]["macro_f1"])
    for k in ("r2", "mae", "rmse", "bias"):
        rows["int_" + k].append(d["intensity_ln"][k]); rows["mag_" + k].append(d["magnitude"][k])
T["base_3seeds"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in rows.items()}

# ---------- T2: effect of the split ----------
lg = load_ev("record_split_ref")["test"]  # earlier checkpoint trained and tested on a record-wise random split
T["split_effect"] = {
    "record_split_earlier_ckpt": {"acc": lg["detection_all"]["accuracy"], "macro_f1": lg["detection_all"]["macro_f1"],
                                  "int_r2": lg["intensity_ln"]["r2"], "int_mae": lg["intensity_ln"]["mae"],
                                  "mag_r2": lg["magnitude"]["r2"], "mag_mae": lg["magnitude"]["mae"],
                                  "thresholds": lg["thresholds_P_S"]},
    "event_split_base_mean": {k: T["base_3seeds"][k][0] for k in ("acc", "macro_f1", "int_r2", "int_mae", "mag_r2", "mag_mae")}}

# ---------- T3: tuned thresholds per model ----------
T["thresholds"] = {os.path.basename(p)[:-5]: load_ev(os.path.basename(p)[:-5])["test"]["thresholds_P_S"]
                   for p in sorted(glob.glob(os.path.join(OUT, "eval", "*.json")))
                   if "_ID" not in p and "record_split" not in p}

# ---------- T4: threshold sensitivity (base seeds) ----------
y = {}; sens = {}
for tag in SEEDS:
    P = preds(tag); st, W, V = chc(tag)
    yt = [int(r["event_type"]) for r in P]; dam = [st[t]["damaging"] == "True" for t in V]
    tuned = tuple(load_ev(tag)["test"]["thresholds_P_S"])
    sens[tag] = {}
    for name, (a, b) in (("tuned", tuned), ("0.50/0.50", (0.5, 0.5)), ("0.64/0.66", (0.64, 0.66)), ("0.3253/0.70", (0.3253, 0.70))):
        yh = [classify(f(r["conf_P"]), f(r["conf_S"]), a, b) for r in P]
        fe = fe_times(st, W, V, a, b)
        sens[tag][name] = {"thr": [a, b], "test_macro_f1": macro_f1(yt, yh),
                           "chc_f1_2s": f1_bin(dam, [fe[t] is not None and fe[t][0] <= 2 for t in V]),
                           "chc_f1_3s": f1_bin(dam, [fe[t] is not None and fe[t][0] <= 3 for t in V])}
T["threshold_sensitivity"] = sens

# ---------- T5: stratified (seed 42) ----------
d = load_ev("base_s42")["test"]
T["by_magnitude"] = {"magnitude": d["magnitude_by_bin"], "intensity": d["intensity_by_mag_bin"]}
P = [r for r in preds("base_s42") if r["event_type"] == "1"]
bins = [(0, 25), (25, 50), (50, 100), (100, 1e9)]; dist = {}
for lo, hi in bins:
    R = [r for r in P if lo <= f(r["epicentral_distance_km"]) < hi]
    ei = np.array([f(r["int_pred_ln"]) - f(r["int_true_ln"]) for r in R]); em = np.array([f(r["mag_pred"]) - f(r["mag_true"]) for r in R])
    dist[f"{lo}-{hi if hi < 1e8 else 'max'}"] = {"n": len(R), "P_recall": sum(r["pred_class"] == "1" for r in R) / len(R),
        "int_mae": float(np.mean(abs(ei))), "int_bias": float(np.mean(ei)), "int_rmse": float(np.sqrt(np.mean(ei ** 2))),
        "mag_mae": float(np.mean(abs(em))), "mag_bias": float(np.mean(em)), "max_km": max(f(r["epicentral_distance_km"]) for r in R)}
T["by_distance"] = dist
T["mmi5_single_window_test"] = {t: load_ev(t)["test"]["mmi5_exceedance_P_windows"] for t in SEEDS}

# ---------- T6: Optuna stages ----------
opt = []
for db in sorted(glob.glob(os.path.join(OUT, "optuna", "*.db"))):
    c = sqlite3.connect(db)
    for sid, name in c.execute("select study_id, study_name from studies"):
        n = c.execute("select count(*) from trials where study_id=? and state='COMPLETE'", (sid,)).fetchone()[0]
        ps = sorted({p for (p,) in c.execute("select distinct param_name from trial_params tp join trials t on tp.trial_id=t.trial_id where t.study_id=?", (sid,))})
        opt.append({"stage": os.path.basename(db)[:-3], "complete_trials": n, "params": ps})
T["optuna"] = opt

# ---------- figures ----------
# S1: example record (needs data/waveforms.hdf5)
sp = {r["event_id"]: r["split"] for r in csv.DictReader(open(SPLIT))}
if os.path.isfile(os.path.join(DATA, "waveforms.hdf5")):
    import h5py
    tr = "2024p051849_MKVS"  # test event, 90 s record, quiet pre-event noise (chosen by pre-event/peak ratio)
    h = h5py.File(os.path.join(DATA, "waveforms.hdf5"), "r")["data"][tr]
    x = np.array(h); fs = float(h.attrs["sampling_rate"]); tp = h.attrs["p_arrival_sample"] / fs; ts = h.attrs["s_arrival_sample"] / fs
    t = np.arange(x.shape[1]) / fs
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 3.6), sharex=True)
    names = h.attrs["channels"].split(",")
    for i in range(3):
        xi = x[i] - np.mean(x[i][: int((tp - 1) * fs)]); xi = xi / np.max(np.abs(xi))
        ax[i].plot(t, xi, color="k", lw=0.4)
        ax[i].set_ylim(-1.1, 1.1); ax[i].set_yticks([-1, 0, 1]); ax[i].set_ylabel(names[i])
        for a0, a1, col in ((tp - 2, tp + 2, C["b"]), (ts - 2, ts + 2, C["o"]), (tp - 6, tp - 2, C["k"]), (tp - 20, tp - 16, C["k"])):
            ax[i].axvspan(a0, a1, color=col, alpha=0.18, lw=0)
        ax[i].axvline(tp, color=C["b"], lw=0.8); ax[i].axvline(ts, color=C["o"], lw=0.8)
    ax[0].text(tp + 0.3, 0.75, "P", color=C["b"]); ax[0].text(ts + 0.3, 0.75, "S", color=C["o"])
    ax[0].text(tp - 4, 0.75, "noise", ha="center", color=C["k"], fontsize=7); ax[0].text(tp - 18, 0.75, "noise", ha="center", color=C["k"], fontsize=7)
    ax[2].set_xlabel("Time from record start (s)"); ax[2].set_xlim(0, t[-1])
    fig.align_ylabels(); fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(FIG, "figS_record.pdf")); plt.close(fig)
    T["example_record"] = {"trace": tr, "magnitude": float(h.attrs["magnitude"]), "distance_km": float(h.attrs["epicentral_distance"]),
                           "split": sp[tr.split("_")[0]], "fs": fs, "tP": tp, "tS": ts}


# S2: distributions
evm = defaultdict(list); recd = defaultdict(list)
for r in csv.DictReader(open(SPLIT)):
    evm[r["split"]].append(float(r["magnitude"]))
for r in csv.DictReader(open(os.path.join(DATA, "metadata.csv"))):
    s_ = sp.get(r["trace_name"].split("_")[0])
    if s_: recd[s_].append(float(r["epicentral_distance"]))
pga = {tag: np.exp(np.array([f(r["int_true_ln"]) for r in preds(tag) if r["event_type"] == "1"]) - 5) for tag in ["base_s42"]}
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
cols = {"train": C["b"], "validation": C["o"], "test": C["g"]}; lab = {"train": "Training", "validation": "Validation", "test": "Test"}
mb = np.arange(3.0, 7.9, 0.2); db_ = np.arange(0, 155, 5)
for s_ in ("train", "validation", "test"):
    ax[0].hist(evm[s_], bins=mb, density=True, histtype="step", color=cols[s_], label=lab[s_])
    ax[1].hist(recd[s_], bins=db_, density=True, histtype="step", color=cols[s_], label=lab[s_])
ax[0].set_yscale("log"); ax[0].set_xlabel("Catalog magnitude"); ax[0].set_ylabel("Density (events)"); ax[0].legend(frameon=False)
ax[1].set_xlabel("Epicentral distance (km)"); ax[1].set_ylabel("Density (records)")
ax[2].hist(pga["base_s42"], bins=np.logspace(-1, 3.2, 40), color=C["g"], alpha=0.7)
ax[2].axvline(42.3, color=C["v"], lw=0.8, ls="--"); ax[2].text(48, ax[2].get_ylim()[1] * 0.4 if False else 300, "MMI 5", color=C["v"], fontsize=7)
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel(r"RotD50 PGA (cm/s$^2$), test P windows"); ax[2].set_ylabel("Windows")
for a, l in zip(ax, "abc"): a.text(-0.22, 1.02, f"({l})", transform=a.transAxes, fontweight="bold")
fig.tight_layout(w_pad=1.0); fig.savefig(os.path.join(FIG, "figS_distributions.pdf")); plt.close(fig)
T["n_events_split"] = {k: len(v) for k, v in evm.items()}; T["n_records_split"] = {k: len(v) for k, v in recd.items()}
T["test_P_windows_above_mmi5"] = int(np.sum(pga["base_s42"] > 42.3))

# S3: training curves
H = lambda tag: json.load(open(os.path.join(OUT, "train", tag + ".json")))
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
for tag, col in zip(SEEDS, (C["b"], C["o"], C["g"])):
    hh = H(tag); ep = np.arange(1, len(hh["history"]["train"]) + 1)
    ax[0].plot(ep, hh["history"]["train"], color=col, lw=0.7, ls=":")
    ax[0].plot(ep, hh["history"]["val"], color=col, lw=0.9, label=f"seed {hh['seed']}")
    ax[0].plot(hh["best_epoch"], hh["best_val_loss"], "o", color=col, ms=3)
ax[0].set_ylim(0.05, 0.4); ax[0].set_yscale("log"); ax[0].set_title("PWaveMamba (dotted: training)", fontsize=7)
for tag, col, lb in (("base_s42", C["b"], "PWaveMamba"), ("mamba_l2_s42", C["s"], "2 Mamba layers"), ("no_pcond_s42", C["p"], "w/o P-cond."),
                     ("lstm_s42", C["v"], "LSTM"), ("transformer_s42", C["o"], "Transformer")):
    hh = H(tag); ax[1].plot(np.arange(1, len(hh["history"]["val"]) + 1), hh["history"]["val"], color=col, lw=0.8, label=lb)
ax[1].set_ylim(0.06, 0.2); ax[1].set_yscale("log"); ax[1].set_title("Validation loss, variants", fontsize=7)
for tag, col, lb in (("base_s42_ID2", C["v"], "1 s"), ("base_s42", C["b"], "2 s"), ("base_s42_ID10", C["o"], "3 s"),
                     ("base_s42_ID13", C["g"], "4 s"), ("base_s42_ID14", C["p"], "5 s")):
    hh = H(tag); ax[2].plot(np.arange(1, len(hh["history"]["val"]) + 1), hh["history"]["val"], color=col, lw=0.8, label=lb)
ax[2].set_ylim(0.05, 0.2); ax[2].set_yscale("log"); ax[2].set_title("Validation loss, post-P length", fontsize=7)
for a, l in zip(ax, "abc"):
    a.set_xlabel("Epoch"); a.legend(frameon=False, fontsize=6); a.text(-0.2, 1.04, f"({l})", transform=a.transAxes, fontweight="bold")
ax[0].set_ylabel("Loss")
fig.tight_layout(w_pad=0.8); fig.savefig(os.path.join(FIG, "figS_training.pdf")); plt.close(fig)
T["training"] = {tag: {"best_epoch": H(tag)["best_epoch"], "epochs_run": H(tag)["epochs_run"], "best_val_loss": H(tag)["best_val_loss"]}
                 for tag in [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(OUT, "train", "*.json")))]}

# S4: Christchurch per station (seed 42, tuned thresholds)
st, W, V = chc("base_s42"); thr = load_ev("base_s42")["test"]["thresholds_P_S"]
fe = fe_times(st, W, V, *thr)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
cnt = defaultdict(int)
for tname in V:
    dkm = f(st[tname]["epicentral_distance_km"]); dmg = st[tname]["damaging"] == "True"; h_ = fe[tname]
    if h_ is None:
        if dmg:
            ax[0].plot(dkm, 6.6, "v", mfc="none", mec=C["v"], ms=4); cnt["FN_none"] += 1
        continue
    lat, mag, _ = h_
    if dmg and lat <= 3: k, mk, col = "TP", "o", C["b"]
    elif dmg: k, mk, col = "FN_late", "o", C["v"]
    elif lat <= 3: k, mk, col = "FP", "s", C["o"]
    else: k, mk, col = "TN_late", "s", C["k"]
    cnt[k] += 1
    ax[0].plot(dkm, min(lat, 6.6), mk, mfc=col if k in ("TP", "FP") else "none", mec=col, ms=3.5, alpha=0.9)
    ax[1].plot(dkm, mag, mk, mfc=col if k in ("TP", "FP") else "none", mec=col, ms=3.5, alpha=0.9)
for D in (2, 3): ax[0].axhline(D, color=C["k"], lw=0.5, ls="--")
ax[0].set_xscale("log"); ax[0].set_xlabel("Epicentral distance (km)"); ax[0].set_ylabel("Alert time after P (s)"); ax[0].set_ylim(0, 7)
ax[1].axhline(6.2, color=C["k"], lw=0.6, ls="--"); ax[1].text(2.6, 6.27, r"M$_\mathrm{w}$ 6.2", fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xlabel("Epicentral distance (km)"); ax[1].set_ylabel("Predicted magnitude at alert")
from matplotlib.lines import Line2D
hd = [Line2D([], [], marker="o", ls="", mfc=C["b"], mec=C["b"], ms=4, label="damaging, alert by 3 s"),
      Line2D([], [], marker="o", ls="", mfc="none", mec=C["v"], ms=4, label="damaging, later alert"),
      Line2D([], [], marker="v", ls="", mfc="none", mec=C["v"], ms=4, label="damaging, no alert"),
      Line2D([], [], marker="s", ls="", mfc=C["o"], mec=C["o"], ms=4, label="not damaging, alert by 3 s"),
      Line2D([], [], marker="s", ls="", mfc="none", mec=C["k"], ms=4, label="not damaging, later alert")]
ax[0].legend(handles=hd, frameon=False, fontsize=6, loc="upper left")
for a, l in zip(ax, "ab"): a.text(-0.15, 1.03, f"({l})", transform=a.transAxes, fontweight="bold")
fig.tight_layout(w_pad=1.5); fig.savefig(os.path.join(FIG, "figS_christchurch.pdf")); plt.close(fig)
T["chc_station_counts_s42"] = dict(cnt)

json.dump(T, open(os.path.join(OUT, "paper", "si_tables.json"), "w"), indent=1)
print(json.dumps({k: T[k] for k in ("base_3seeds", "split_effect", "threshold_sensitivity", "by_distance", "mmi5_single_window_test",
                                     "n_events_split", "n_records_split", "test_P_windows_above_mmi5", "chc_station_counts_s42")}, indent=0)[:6000])
