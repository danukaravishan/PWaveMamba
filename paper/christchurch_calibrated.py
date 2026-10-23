#!/usr/bin/env python3
"""Christchurch and NZ-test comparison with thresholds fixed on NZ data for every method (Sections VI-E, VII, Fig. 3).

    python3 paper/christchurch_calibrated.py results       # released outputs
    python3 paper/christchurch_calibrated.py outputs       # your own re-run

* PWaveMamba / LSTM / Transformer (3 seeds each): alert level L chosen on the NZ VALIDATION P windows
  (single window, 1.5-2.5 s of P wave) as the PGA level that maximizes the F1 of "PGA > 42.3 cm/s^2";
  the first-exceedance rule then alerts when a P-classified window predicts PGA > L.
* P_d: threshold chosen on the NZ TRAINING records (pd_calibration.py, max F1, per window length tau).
* Also: MMI-5 level (L = 42.3) and P_d > 0.5 cm, station ROC AUC by deadline, and the NZ test set comparison
  (single window / single record exceedance F1 and AUC on the same 7,193 P records).
Nothing is chosen on Christchurch. Output: <dir>/paper/christchurch_calibrated.json
"""
import csv, gzip, json, math, os, sys
from collections import defaultdict
import numpy as np

OUT = sys.argv[1]
C = os.path.join(OUT, "christchurch")
TAU = math.log(42.3) + 5
f = lambda x: float(x) if x not in ("", "nan", "None", "NaN", None) else float("nan")
DS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
FAM = {"PWaveMamba": ["base_s42", "base_s7", "base_s123"], "LSTM": ["lstm_s42", "lstm_s7", "lstm_s123"],
       "Transformer": ["transformer_s42", "transformer_s7", "transformer_s123"]}


def conf(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "TP": tp, "FP": fp, "FN": fn}


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); pos, neg = s[y], s[~y]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def read_preds(path):
    if not os.path.isfile(path) and os.path.isfile(path + ".gz"):
        path += ".gz"
    op = gzip.open if path.endswith(".gz") else open
    return [r for r in csv.DictReader(op(path, "rt")) if r["event_type"] == "1"]


def best_level(P):
    yt = np.array([f(r["int_true_ln"]) > TAU for r in P]); sp = np.array([f(r["int_pred_ln"]) for r in P])
    lv = np.arange(10, 60.01, 0.5)
    f1s = [conf(yt, sp > math.log(L) + 5)["f1"] for L in lv]
    return float(lv[int(np.argmax(f1s))]), float(max(f1s))


st = {r["trace_name"]: r for r in csv.DictReader(open(os.path.join(C, "base_s42_stations.csv")))}
V = [t for t, r in st.items() if not math.isnan(f(r["pga_meta_cms2"])) and not math.isnan(f(r["tP"]))]
y = np.array([st[t]["damaging"] == "True" for t in V])
res = {"n": len(V), "n_damaging": int(y.sum()), "models": {}, "Pd": {}, "nz_test": {}}


def scores(tag):
    W = defaultdict(list)
    for r in csv.DictReader(gzip.open(os.path.join(C, tag + "_windows.csv.gz"), "rt")):
        W[r["trace_name"]].append((f(r["window_end_s"]), int(r["class"]), f(r["int_pred_ln"])))
    out = {}
    for D in DS:
        s = []
        for t in V:
            tP = f(st[t]["tP"])
            v = [i for e, c, i in W[t] if c == 1 and e <= tP + D + 1e-9]
            s.append(max(v) if v else -np.inf)
        out[D] = np.array(s)
    return out


for fam, tags in FAM.items():
    for tag in tags:
        Lv, f1v = best_level(read_preds(os.path.join(OUT, "eval", tag + "_validation_predictions.csv.gz")))
        Pt = read_preds(os.path.join(OUT, "eval", tag + "_test_predictions.csv"))
        yt = np.array([f(r["int_true_ln"]) > TAU for r in Pt]); sp = np.array([f(r["int_pred_ln"]) for r in Pt])
        S = scores(tag)
        res["models"][tag] = {"family": fam, "level_val_cms2": Lv, "val_F1": f1v,
                              "nz_test": {"AUC": auc(yt, sp), "F1_mmi5": conf(yt, sp > TAU)["f1"], "F1_level": conf(yt, sp > math.log(Lv) + 5)["f1"]},
                              "chc": {str(D): {"AUC": auc(y, S[D]), "mmi5": conf(y, S[D] > TAU), "level": conf(y, S[D] > math.log(Lv) + 5)} for D in DS}}

pdc = json.load(open(os.path.join(OUT, "pd", "pd_calibration.json")))
pdr = list(csv.DictReader(open(os.path.join(OUT, "pd", "christchurch_pd.csv"))))
pdv = {r["trace_name"]: r for r in pdr}
for k in ("1", "1.5", "2", "2.5", "3"):
    s = np.array([f(pdv[t][f"Pd{k}_cm"]) if t in pdv else np.nan for t in V]); s = np.where(np.isfinite(s), s, -np.inf)
    thr = pdc["by_tau"][k]["thr_trainF1_cm"]
    res["Pd"][k] = {"thr_NZ_cm": thr, "AUC": auc(y, s), "at_0.5cm": conf(y, s > 0.5), "at_NZ_thr": conf(y, s > thr),
                    "nz_test": {"AUC": pdc["by_tau"][k]["test"]["AUC"], "F1_0.5cm": pdc["by_tau"][k]["test"]["at_0.5cm"]["f1"],
                                "F1_NZ_thr": pdc["by_tau"][k]["test"]["at_trainF1_thr"]["f1"]}}
from math import comb
def mcnemar(a, b):
    a, b = np.asarray(a, bool) == y, np.asarray(b, bool) == y
    n1, n2 = int((a & ~b).sum()), int((~a & b).sum()); n = n1 + n2
    return {"model_right_pd_wrong": n1, "model_wrong_pd_right": n2,
            "p": min(1.0, 2 * sum(comb(n, k) for k in range(min(n1, n2) + 1)) / 2 ** n) if n else 1.0}
res["mcnemar_level_vs_PdNZ"] = {}
for tag in FAM["PWaveMamba"]:
    S = scores(tag); Lv = res["models"][tag]["level_val_cms2"]
    for D in ("2", "3"):
        s = np.array([f(pdv[t][f"Pd{D}_cm"]) if t in pdv else np.nan for t in V]); s = np.where(np.isfinite(s), s, -np.inf)
        res["mcnemar_level_vs_PdNZ"][f"{tag}_D{D}"] = mcnemar(S[float(D)] > math.log(Lv) + 5, s > res["Pd"][D]["thr_NZ_cm"])
os.makedirs(os.path.join(OUT, "paper"), exist_ok=True)
json.dump(res, open(os.path.join(OUT, "paper", "christchurch_calibrated.json"), "w"), indent=1)

# ---------------- print
print("McNemar (Mamba at NZ level vs P_d at NZ threshold):", {k: (v["model_right_pd_wrong"], v["model_wrong_pd_right"], round(v["p"], 3)) for k, v in res["mcnemar_level_vs_PdNZ"].items()})
print(f"Christchurch: {res['n']} stations, {res['n_damaging']} damaging")
print("\nNZ-validation alert levels (cm/s^2) and NZ test (P windows / records, 29 above MMI 5):")
for tag, m in res["models"].items():
    print(f"  {tag:17s} L={m['level_val_cms2']:5.1f}  test AUC {m['nz_test']['AUC']:.3f}  F1 mmi5 {m['nz_test']['F1_mmi5']:.2f}  F1 L {m['nz_test']['F1_level']:.2f}")
for k, v in res["Pd"].items():
    print(f"  Pd tau={k:3s} thr_NZ={v['thr_NZ_cm']:.4f} cm  test AUC {v['nz_test']['AUC']:.3f}  F1 0.5cm {v['nz_test']['F1_0.5cm']:.2f}  F1 NZthr {v['nz_test']['F1_NZ_thr']:.2f}")
print("\nChristchurch F1 by deadline D (seed mean [min-max]) and AUC mean:")
for fam, tags in FAM.items():
    for rule in ("mmi5", "level"):
        row = []
        for D in DS:
            v = [res["models"][t]["chc"][str(D)][rule]["f1"] for t in tags]
            row.append(f"{np.mean(v):.2f}[{min(v):.2f}-{max(v):.2f}]")
        print(f"  {fam:11s} {rule:5s} " + " ".join(f"D{D:g}:{r}" for D, r in zip(DS, row)))
    print(f"  {fam:11s} AUC   " + " ".join(f"D{D:g}:{np.mean([res['models'][t]['chc'][str(D)]['AUC'] for t in tags]):.3f}" for D in DS))
for k, v in res["Pd"].items():
    print(f"  Pd D{k:3s}: AUC {v['AUC']:.3f}  F1@0.5cm {v['at_0.5cm']['f1']:.2f}  F1@NZthr {v['at_NZ_thr']['f1']:.2f} ({v['at_NZ_thr']['TP']}/{v['at_NZ_thr']['FP']}/{v['at_NZ_thr']['FN']})")
