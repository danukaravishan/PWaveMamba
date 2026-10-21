#!/usr/bin/env python3
"""Decision rules on the saved Christchurch per-window outputs (stdlib only; runs anywhere).

    python3 paper/christchurch_rules.py results            # released outputs
    python3 paper/christchurch_rules.py outputs            # your own re-run

Rules (all causal: a decision at time t uses only windows ending <= t):
  FE(D) : first exceedance. Alert at the first window that is classified P and predicts
          y_I > tau (MMI 5, ln(42.3)+5); the alert counts if issued by t_P + D.
  HOLD(H): first P window + hold H, max intensity of the P windows ending by then.
  Pd(tau): from the station CSV (2 s, 3 s).
Reads <dir>/christchurch/*_windows.csv.gz, *_stations.csv and <dir>/eval/*_test_predictions.csv(.gz).
Outputs <dir>/paper/christchurch_rules.json and christchurch_tradeoff.csv (Fig. 3 and Table III F1 columns).
"""
import csv, gzip, json, math, os, random, sys
from collections import defaultdict
from math import comb

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
DST = os.path.join(OUT, "paper")
TAU = math.log(42.3) + 5.0
DS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 1e9]
f = lambda x: float(x) if x not in ("", "nan", "None", "NaN") else float("nan")


def conf(y, p):
    tp = sum(a and b for a, b in zip(y, p)); fp = sum((not a) and b for a, b in zip(y, p))
    fn = sum(a and not b for a, b in zip(y, p)); tn = len(y) - tp - fp - fn
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0}


def boot(y, p, n=5000, seed=0):
    r = random.Random(seed); idx = range(len(y)); v = []
    for _ in range(n):
        s = [r.choice(idx) for _ in idx]
        v.append(conf([y[i] for i in s], [p[i] for i in s])["f1"])
    v.sort(); return [round(v[int(0.025 * n)], 3), round(v[int(0.975 * n) - 1], 3)]


def mcnemar(y, a, b):
    b1 = sum((x == t) and (z != t) for x, z, t in zip(a, b, y)); c1 = sum((x != t) and (z == t) for x, z, t in zip(a, b, y))
    n = b1 + c1
    return {"a_right_b_wrong": b1, "a_wrong_b_right": c1,
            "p": min(1.0, 2 * sum(comb(n, k) for k in range(min(b1, c1) + 1)) / 2 ** n) if n else 1.0}


res, trade = {}, []
cdir = os.path.join(OUT, "christchurch")
for fn in sorted(os.listdir(cdir)):
    if not fn.endswith("_windows.csv.gz"):
        continue
    tag = fn[:-len("_windows.csv.gz")]
    st = {r["trace_name"]: r for r in csv.DictReader(open(os.path.join(cdir, tag + "_stations.csv")))}
    W = defaultdict(list)
    for r in csv.DictReader(gzip.open(os.path.join(cdir, fn), "rt")):
        W[r["trace_name"]].append((f(r["window_end_s"]), int(r["class"]), f(r["int_pred_ln"])))
    V = [t for t, r in st.items() if not math.isnan(f(r["pga_meta_cms2"])) and not math.isnan(f(r["tP"]))]
    y = [st[t]["damaging"] == "True" for t in V]
    fe_t, rows = {}, []
    for t in V:
        tP = f(st[t]["tP"])
        ex = [e for e, c, i in sorted(W[t]) if c == 1 and i > TAU]
        fe_t[t] = ex[0] - tP if ex else float("inf")
    out = {"n": len(V), "n_damaging": sum(y), "tau_ln": TAU, "FE": {}, "Pd": {}}
    for D in DS:
        p = [fe_t[t] <= D for t in V]
        c = conf(y, p); c["f1_ci95"] = boot(y, p)
        key = "inf" if D > 1e8 else f"{D:g}"
        out["FE"][key] = c
        trade.append([tag, "FE", key, c["TP"], c["FP"], c["FN"], c["TN"], round(c["f1"], 4)])
    for tau in ("2", "3"):
        p = [st[t][f"alert_Pd{tau}"] == "True" for t in V]
        c = conf(y, p); c["f1_ci95"] = boot(y, p); out["Pd"][tau] = c
        trade.append([tag, "Pd", tau, c["TP"], c["FP"], c["FN"], c["TN"], round(c["f1"], 4)])
    pd3 = [st[t]["alert_Pd3"] == "True" for t in V]; pd2 = [st[t]["alert_Pd2"] == "True" for t in V]
    out["mcnemar"] = {f"FE{D}_vs_Pd{tau}": mcnemar(y, [fe_t[t] <= D for t in V], pdv)
                      for D in (2.0, 3.0) for tau, pdv in (("2", pd2), ("3", pd3))}
    tp_lat = sorted(fe_t[t] for t, yy in zip(V, y) if yy and fe_t[t] < 1e8)
    lead = sorted(f(st[t]["t_PGA"]) - (f(st[t]["tP"]) + fe_t[t]) for t, yy in zip(V, y) if yy and fe_t[t] < 1e8)
    q = lambda a, x: a[min(len(a) - 1, int(x * len(a)))] if a else None
    out["FE_latency_TP"] = {"n": len(tp_lat), "median": q(tp_lat, .5), "q25": q(tp_lat, .25), "q75": q(tp_lat, .75),
                            "min": tp_lat[0] if tp_lat else None}
    out["FE_lead_to_PGA_TP"] = {"median": q(lead, .5), "frac_positive": sum(x > 0 for x in lead) / len(lead) if lead else None}
    out["FE_alerts_before_P"] = sum(1 for t in V if fe_t[t] < 0)
    res[tag] = out

# false alarms of FE on test-set noise windows (single-window proxy)
fa = {}
for fn in sorted(os.listdir(os.path.join(OUT, "eval"))):
    if fn.endswith(("_test_predictions.csv", "_test_predictions.csv.gz")):
        n = k = 0
        p = os.path.join(OUT, "eval", fn)
        for r in csv.DictReader(gzip.open(p, "rt") if fn.endswith(".gz") else open(p)):
            if r["event_type"] == "0":
                n += 1; k += (r["pred_class"] == "1" and f(r["int_pred_ln"]) > TAU)
        fa[fn.split("_test_predictions.csv")[0]] = {"noise_windows": n, "classified_P_and_above_tau": k}
res["_test_noise_false_alarms"] = fa
os.makedirs(DST, exist_ok=True)
json.dump(res, open(os.path.join(DST, "christchurch_rules.json"), "w"), indent=1)
with open(os.path.join(DST, "christchurch_tradeoff.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["model", "rule", "D_or_tau_s", "TP", "FP", "FN", "TN", "F1"]); w.writerows(trade)
for tag, o in res.items():
    if tag.startswith("_"):
        continue
    print(f"\n{tag}: n={o['n']} dam={o['n_damaging']} alerts_before_P={o['FE_alerts_before_P']}  latency(TP)={o['FE_latency_TP']}  lead={o['FE_lead_to_PGA_TP']}")
    print("  FE F1 by D:", {k: (round(v['f1'], 2), v['TP'], v['FP']) for k, v in o["FE"].items()})
    print("  Pd:", {k: round(v['f1'], 2) for k, v in o["Pd"].items()}, " FE3 CI", o["FE"]["3"]["f1_ci95"])
    print("  McNemar:", {k: round(v['p'], 3) for k, v in o["mcnemar"].items()})
print("\nnoise false alarms:", fa)
