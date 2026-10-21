#!/usr/bin/env python3
"""Print the numbers of Tables I, III, IV and V of the paper from the saved outputs (standard library only).

    python3 paper/christchurch_rules.py results      # once, writes results/paper/christchurch_rules.json
    python3 paper/main_tables.py results

Use `outputs` instead of `results` to tabulate your own re-run.
"""
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
SEEDS = ["base_s42", "base_s7", "base_s123"]


def ev(tag):
    return json.load(open(os.path.join(R, "eval", tag + ".json")))


def line(*cols):
    print("  ".join(f"{c:>12}" if i else f"{c:<22}" for i, c in enumerate(cols)))


# ---------------- Table I: dataset and split
summ_path = os.path.join(R, "..", "splits", "event_split_summary.json")
if not os.path.isfile(summ_path):
    summ_path = os.path.join(ROOT, "splits", "event_split_summary.json")
S = json.load(open(summ_path))["per_split"]
print("\nTable I  Dataset and event-disjoint split")
line("", "train", "validation", "test")
line("events", *[S[s]["events"] for s in ("train", "validation", "test")])
line("records", *[S[s]["records"] for s in ("train", "validation", "test")])
for b in ("3-4", "4-5", "5-6", "6-99"):
    line(f"events M {b}", *[S[s]["events_by_bin"][b] for s in ("train", "validation", "test")])
line("max magnitude", *[f"{S[s]['max_magnitude']:.2f}" for s in ("train", "validation", "test")])
cm = ev("base_s42")["test"]["detection_all"]["confusion"]
print(f"test windows used: {sum(map(sum, cm))} (noise/P/S = {', '.join(str(sum(r)) for r in cm)})")

# ---------------- Table III: ablations and baselines
rules_path = os.path.join(R, "paper", "christchurch_rules.json")
rules = json.load(open(rules_path)) if os.path.isfile(rules_path) else {}


def f1(tag, D):
    return rules[tag]["FE"][D]["f1"] if tag in rules else float("nan")


print("\nTable III  Ablations and baselines (test set; F1 = Christchurch first-exceedance alerts by 2 / 3 s)")
line("model", "params", "acc", "int R2", "mag R2", "mag MAE", "F1 2s", "F1 3s")
T = [ev(t)["test"] for t in SEEDS]
m = lambda f: st.mean(f(t) for t in T)
line("PWaveMamba (3 seeds)", ev("base_s42")["params"], f"{m(lambda t: t['detection_all']['accuracy']):.3f}",
     f"{m(lambda t: t['intensity_ln']['r2']):.3f}+-{st.stdev(t['intensity_ln']['r2'] for t in T):.3f}",
     f"{m(lambda t: t['magnitude']['r2']):.3f}", f"{m(lambda t: t['magnitude']['mae']):.3f}",
     f"{st.mean(f1(s, '2') for s in SEEDS):.2f}", f"{st.mean(f1(s, '3') for s in SEEDS):.2f}")
for tag, name in (("mamba_l2_s42", "2 Mamba layers"), ("no_pcond_s42", "w/o P-cond. pooling"),
                  ("det_only_s42", "detection only"), ("int_only_s42", "intensity only"),
                  ("mag_only_s42", "magnitude only"), ("lstm_s42", "LSTM backbone"),
                  ("transformer_s42", "Causal Transformer")):
    e = ev(tag)
    t = e["test"]
    acc = f"{t['detection_all']['accuracy']:.3f}" if tag not in ("int_only_s42", "mag_only_s42") else "--"
    ir2 = f"{t['intensity_ln']['r2']:.3f}" if tag not in ("det_only_s42", "mag_only_s42") else "--"
    mr2 = f"{t['magnitude']['r2']:.3f}" if tag not in ("det_only_s42", "int_only_s42") else "--"
    mmae = f"{t['magnitude']['mae']:.3f}" if tag not in ("det_only_s42", "int_only_s42") else "--"
    single = tag in ("det_only_s42", "int_only_s42", "mag_only_s42")
    line(name, e["params"], acc, ir2, mr2, mmae, "--" if single else f"{f1(tag, '2'):.2f}",
         "--" if single else f"{f1(tag, '3'):.2f}")

# ---------------- Table IV: compute
for p in sorted(glob.glob(os.path.join(R, "bench", "*.json"))):
    B = json.load(open(p))
    print(f"\nTable IV  Cost per window ({os.path.basename(p)}: {B.get('env', {}).get('gpu')}, {B.get('cpu')})")
    line("model", "params", "GPU b1 ms", "GPU win/s b64", "CPU 1 thr ms")
    for v, e in B["models"].items():
        cpu = e.get("cpu_torch_t1", {}).get("median_ms")
        if cpu is None and v == "base":
            cpu = B.get("mamba_pure_torchscript", {}).get("cpu_t1", {}).get("median_ms")
        line(v, e.get("params"), f"{e['gpu_b1']['median_ms']:.2f}" if "gpu_b1" in e else "--",
             f"{e['gpu_windows_per_s_b64']:.0f}" if "gpu_windows_per_s_b64" in e else "--",
             f"{cpu:.1f}" if cpu else "--")
    macs = B.get("analytic_macs_base") or B.get("analytic_macs", {}).get("base")
    if macs:
        print(f"analytic MACs, PWaveMamba: {macs['total'] / 1e6:.1f} M")

# ---------------- Table V: window length
print("\nTable V  Post-P window length (one seed, test set)")
line("post-P (s)", "acc", "int R2", "int MAE", "mag R2", "mag MAE", "M>=6 bias mag", "M>=6 bias int")
for tag, post in (("base_s42_ID2", 1), ("base_s42", 2), ("base_s42_ID10", 3), ("base_s42_ID13", 4), ("base_s42_ID14", 5)):
    t = ev(tag)["test"]
    line(post, f"{t['detection_all']['accuracy']:.3f}", f"{t['intensity_ln']['r2']:.3f}", f"{t['intensity_ln']['mae']:.3f}",
         f"{t['magnitude']['r2']:.3f}", f"{t['magnitude']['mae']:.3f}", f"{t['magnitude_by_bin']['6-9']['bias']:.2f}",
         f"{t['intensity_by_mag_bin']['6-9']['bias']:.2f}")
