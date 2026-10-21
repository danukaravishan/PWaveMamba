#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data (CPU, pure-PyTorch Mamba; about 5 minutes).

    python tests/smoke_test.py

Builds a tiny fake data set, then runs segment extraction, the split, one training epoch, evaluation,
the Christchurch pipeline and the decision-rule analysis, and checks that every released checkpoint
loads with strict=True. The numbers it produces are meaningless; it only checks that the code runs.
"""
import csv
import glob
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
ENV = dict(os.environ, PWAVEMAMBA_WINDOW="ID6")


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True, cwd=ROOT, env=ENV)


def fake_record(rng, n=9000, ip=3000, isx=3600, amp=0.05):
    t = np.arange(n)
    x = rng.normal(0, 1e-4, (3, n))
    env_p = np.where(t >= ip, np.exp(-(t - ip) / 300.0), 0)
    env_s = np.where(t >= isx, np.exp(-(t - isx) / 400.0), 0)
    for c in range(3):
        x[c] += amp * 0.3 * env_p * np.sin(2 * np.pi * 8 * t / 100 + c) + amp * env_s * np.sin(2 * np.pi * 3 * t / 100 + c)
    return x


def make_data(d):
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(d, "christchurch"))
    rows = []
    with h5py.File(os.path.join(d, "waveforms.hdf5"), "w") as f:
        g = f.create_group("data")
        for e in range(20):
            ev = f"2020p{e:06d}"
            mag = 3.0 + 3.5 * (e % 10) / 10
            for st in ("AAA", "BBB"):
                tr = f"{ev}_{st}"
                ds = g.create_dataset(tr, data=fake_record(rng, amp=0.01 * mag))
                ds.attrs.update({"sampling_rate": 100, "channels": "HNE,HNN,HNZ", "magnitude": mag,
                                 "p_arrival_sample": 3000, "s_arrival_sample": 3600, "epicentral_distance": 20 + e})
                rows.append({"trace_name": tr, "magnitude": mag, "new_p_arrival_sample": 3000, "new_s_arrival_sample": 3600,
                             "epicentral_distance": 20 + e, "event_depth": 10, "station": st, "magnitude_type": "ML"})
    with open(os.path.join(d, "metadata.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    crows = []
    with h5py.File(os.path.join(d, "christchurch", "christchurch.hdf5"), "w") as f:
        g = f.create_group("data")
        for i in range(6):
            tr = f"2011p150000_C{i:02d}"
            g.create_dataset(tr, data=fake_record(rng, n=3000, ip=1000, isx=1400, amp=0.2 * (i + 1))).attrs.update(
                {"sampling_rate": 100, "channels": "HNE,HNN,HNZ"})
            crows.append({"trace_name": tr, "station": f"C{i:02d}", "epicentral_distance": 5 + 10 * i, "sampling_rate": 100,
                          "p_arrival_sample": 1000, "s_arrival_sample": 1400, "pga_rotd50": 0.1 * (i + 1),
                          "trace_starttime": "2011-02-21T23:51:00Z"})
    with open(os.path.join(d, "christchurch", "metadata.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crows[0]))
        w.writeheader()
        w.writerows(crows)


def main():
    import torch
    from pwavemamba import build_model, load_checkpoint
    from pwavemamba.config import MODEL_HPARAMS  # noqa: F401

    # 1. released checkpoints load into the reference implementation
    for ck in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*.pt"))):
        v = torch.load(ck, map_location="cpu", weights_only=False).get("variant", "base")
        m = build_model(v, backend="reference")
        load_checkpoint(m, ck)
        out = m.eval()(torch.zeros(2, 3, 400))
        assert out[2].shape == (2, 400, 3)
        print(f"ok  {os.path.basename(ck):24s} {v:12s} {sum(p.numel() for p in m.parameters()):,} params")

    # 2. pipeline on synthetic data
    tmp = tempfile.mkdtemp(prefix="pwavemamba_smoke_")
    data, out = os.path.join(tmp, "data"), os.path.join(tmp, "outputs")
    make_data(data)
    run("scripts/extract_segments.py", "--data-dir", data)
    run("scripts/make_split.py", "--metadata", os.path.join(data, "metadata.csv"), "--out-dir", os.path.join(tmp, "splits"))
    split = os.path.join(tmp, "splits", "event_split.csv")
    common = ["--data-dir", data, "--split-csv", split, "--out", out]
    run("scripts/train.py", "--variant", "base", "--seed", "1", "--epochs", "1", "--batch", "4", "--backend", "reference", *common)
    ck = os.path.join(out, "models", "base_s1.pt")
    run("scripts/evaluate.py", "--ckpt", ck, "--backend", "reference", *common)
    run("scripts/christchurch_eval.py", "--ckpt", ck, "--eval-json", os.path.join(out, "eval", "base_s1.json"),
        "--hdf5", os.path.join(data, "christchurch", "christchurch.hdf5"),
        "--meta", os.path.join(data, "christchurch", "metadata.csv"), "--out", out, "--stride", "0.5",
        "--backend", "reference")
    run("paper/christchurch_rules.py", out)
    print(f"\nSMOKE TEST PASSED (scratch files in {tmp})")


if __name__ == "__main__":
    main()
