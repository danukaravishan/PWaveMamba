#!/usr/bin/env python3
"""Figures 2 and 3 of the article, from the saved run outputs.

    python3 paper/fig_paper.py results       # writes results/paper/figures/fig_test.pdf, fig_tradeoff.pdf

Fig. 2 (fig_test.pdf): test-set performance of PWaveMamba, seed 42
  (a) PR curves of the window-level P and S decisions   <- eval/base_s42.json  (pr_curve_P, pr_curve_S)
  (b) row-normalized confusion matrix                    <- eval/base_s42.json  (detection_all.confusion)
  (c) predicted vs observed RotD50 PGA, P windows        <- eval/base_s42_test_predictions.csv(.gz)
  (d) predicted vs catalog magnitude, P windows          <- same
Fig. 3 (fig_tradeoff.pdf): Christchurch F1 of MMI 5 alerts vs decision deadline D, 3 seeds per
  backbone, (a) MMI-5 level / P_d > 0.5 cm, (b) thresholds fixed on NZ data
  <- <dir>/paper/christchurch_calibrated.json written by paper/christchurch_calibrated.py.
Needs numpy and matplotlib.
"""
import csv, gzip, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(OUT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
LN10 = math.log(10.0)
C = dict(b="#0072B2", o="#E69F00", g="#009E73", v="#D55E00", k="#333333")
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42})


def find(*names):
    for n in names:
        if os.path.isfile(n):
            return n
    raise FileNotFoundError(names[0])


def preds(tag):
    p = find(os.path.join(OUT, "eval", tag + "_test_predictions.csv"), os.path.join(OUT, "eval", tag + "_test_predictions.csv.gz"))
    return list(csv.DictReader(gzip.open(p, "rt") if p.endswith(".gz") else open(p)))


def r2(t, p):
    return 1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)


# ------------------------------------------------------------------ Fig. 2
ev = json.load(open(os.path.join(OUT, "eval", "base_s42.json")))["test"]
P = [r for r in preds("base_s42") if r["event_type"] == "1"]
it = np.array([float(r["int_true_ln"]) for r in P]); ip = np.array([float(r["int_pred_ln"]) for r in P])
mt = np.array([float(r["mag_true"]) for r in P]); mp = np.array([float(r["mag_pred"]) for r in P])
lt, lp = (it - 5) / LN10, (ip - 5) / LN10                      # log10 PGA [cm/s^2]

fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.95), gridspec_kw={"width_ratios": [1.15, 0.85, 1, 1]})
for key, col, mk, lab in (("pr_curve_P", C["b"], "o", "P wave"), ("pr_curve_S", C["v"], "s", "S wave")):
    c = np.array(ev[key]); ax[0].plot(c[:, 2], c[:, 1], marker=mk, ms=2, color=col, label=lab)
ax[0].set_xlim(0.9, 1.0); ax[0].set_ylim(0.9, 1.0); ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
ax[0].legend(frameon=True, loc="lower left"); ax[0].grid(alpha=0.3, lw=0.4)

cm = np.array(ev["detection_all"]["confusion"]); cn = cm / cm.sum(1, keepdims=True)
ax[1].imshow(cn, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax[1].text(j, i, f"{cn[i, j]:.3f}\n({cm[i, j]})", ha="center", va="center", fontsize=6,
                   color="white" if cn[i, j] > 0.5 else "black")
lab3 = ["Noise", "P", "S"]
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(lab3); ax[1].set_yticks(range(3)); ax[1].set_yticklabels(lab3)
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True")

thr = math.log10(42.3)
for a, x, y, txt, xl, yl in ((ax[2], lt, lp, f"$R^2$ = {r2(it, ip):.3f}", r"Observed log$_{10}$ PGA (cm/s$^2$)", r"Predicted log$_{10}$ PGA (cm/s$^2$)"),
                             (ax[3], mt, mp, f"$R^2$ = {r2(mt, mp):.3f}\nMAE = {np.mean(np.abs(mp - mt)):.3f}", "Catalog magnitude", "Predicted magnitude")):
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    a.hexbin(x, y, gridsize=45, cmap="Blues", norm=LogNorm(), mincnt=1, linewidths=0.1)
    a.plot([lo, hi], [lo, hi], color=C["k"], lw=0.7)
    a.text(0.04, 0.96, txt, transform=a.transAxes, va="top", fontsize=7)
    a.set_xlabel(xl); a.set_ylabel(yl)
ax[2].axvline(thr, color=C["k"], ls=":", lw=0.8); ax[2].axhline(thr, color=C["k"], ls=":", lw=0.8)
for a, l in zip(ax, "abcd"):
    a.set_title(f"({l})", loc="left", fontsize=8, pad=2)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(FIG, "fig_test.pdf")); plt.close(fig)

# ------------------------------------------------------------------ Fig. 3
B3 = json.load(open(os.path.join(OUT, "paper", "christchurch_calibrated.json")))
FAMS = (("PWaveMamba", ["base_s42", "base_s7", "base_s123"], C["b"], "o"), ("Transformer", ["transformer_s42", "transformer_s7", "transformer_s123"], C["v"], "^"),
        ("LSTM", ["lstm_s42", "lstm_s7", "lstm_s123"], C["g"], "x"))
Dk = sorted(B3["models"]["base_s42"]["chc"], key=float); D = np.array([float(k) for k in Dk])
Pk = sorted(B3["Pd"], key=float); PD = np.array([float(k) for k in Pk])
fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.1), sharex=True)
for ax, rule, pdkey, pdlab in ((axs[0], "mmi5", "at_0.5cm", r"$P_d > 0.5$ cm"), (axs[1], "level", "at_NZ_thr", r"$P_d$, NZ-calibrated threshold")):
    for fam, tags, col, mk in FAMS:
        Y = np.array([[B3["models"][t]["chc"][k][rule]["f1"] for k in Dk] for t in tags])
        ax.fill_between(D, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(D, Y.mean(0), marker=mk, color=col, ms=3, lw=1.0, label=fam + " (3 seeds)")
    ax.plot(PD, [B3["Pd"][k][pdkey]["f1"] for k in Pk], "D--", color=C["k"], ms=3.5, lw=0.9, label=pdlab)
    ax.set_ylim(0, 1.0); ax.set_xlim(0.4, 6.2); ax.grid(alpha=0.3, lw=0.4); ax.set_ylabel("Alert F1 score")
    ax.legend(loc="lower right", fontsize=6, frameon=True)
axs[0].set_title("(a) MMI 5 level, no calibration", loc="left", fontsize=8, pad=2)
axs[1].set_title("(b) Thresholds fixed on New Zealand data", loc="left", fontsize=8, pad=2)
axs[1].set_xlabel("Decision deadline after P arrival, $D$ (s)")
fig.tight_layout(h_pad=0.6)
fig.savefig(os.path.join(FIG, "fig_tradeoff.pdf")); plt.close(fig)
Y = np.array([[B3["models"][t]["chc"][k]["mmi5"]["f1"] for k in Dk] for t in FAMS[0][1]])
print("wrote", os.path.join(FIG, "fig_test.pdf"), "and fig_tradeoff.pdf")
print("Fig.2 R2 int %.4f, mag R2 %.4f MAE %.4f; Fig.3 PWaveMamba mean F1@2/3 s %.3f/%.3f" % (
    r2(it, ip), r2(mt, mp), np.mean(np.abs(mp - mt)), Y.mean(0)[list(D).index(2.0)], Y.mean(0)[list(D).index(3.0)]))
