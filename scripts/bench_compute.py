"""Complexity and latency per 4 s window (Table IV of the paper).

    python scripts/bench_compute.py                # GPU (mamba_ssm kernels) + CPU
    python scripts/bench_compute.py --cpu-only

For each backbone: parameters, analytic MACs (Mamba) or ptflops MACs (if installed), latency at batch 1
(median and p95 of N runs) and GPU throughput at batch 64. On the CPU the Mamba models run the
pure-PyTorch reference scan (pwavemamba/mamba_ref.py) as a TorchScript trace, and through ONNX Runtime if
`onnx` and `onnxruntime` are installed. Latency does not depend on the trained weights, so random weights
are used. Output: <out>/bench/<host>[_cpu].json
"""
import json
import os
import platform
import socket
import statistics
import tempfile
import time

import _common
import torch

from pwavemamba.config import MODEL_HPARAMS
from pwavemamba.utils import env_info, save_json
from pwavemamba.variants import build_model

ap = _common.base_parser(__doc__.split("\n")[0])
ap.add_argument("--cpu-only", action="store_true")
ap.add_argument("--runs", type=int, default=500)
ap.add_argument("--threads", type=int, nargs="*", default=[1, 4])
a = ap.parse_args()


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def timeit(fn, runs, sync=None, warm=30):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(runs):
        if sync:
            sync()
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return {"median_ms": statistics.median(ts), "p95_ms": ts[int(0.95 * len(ts)) - 1], "runs": runs}


def analytic_macs(n_layers=4, L=400):
    """Multiply-accumulates of one window: stem, Mamba layers (projections, conv, scan) and heads."""
    n = MODEL_HPARAMS
    d, E, N, K = n["d_model"], n["expand"], n["d_state"], n["d_conv"]
    di, r = E * d, -(-d // 16)
    stem = L * (3 * n["stem_conv1_filters"] * n["stem_kernel_size1"]
                + n["stem_conv1_filters"] * n["stem_conv2_filters"] * n["stem_kernel_size2"]
                + n["stem_conv2_filters"] * n["stem_conv3_filters"] * n["stem_kernel_size3"]
                + n["stem_conv3_filters"] * d)
    per_step = d * 2 * di + di * K + di * (r + 2 * N) + r * di + 3 * di * N + di * d
    det = L * (2 * d * 32 + 32 * 8 + 8 * 3)
    heads = 2 * (2 * d * n["fc1_neurons"] + n["fc1_neurons"] * n["fc2_neurons"] + n["fc2_neurons"]) + L * d
    return {"stem": stem, "backbone": n_layers * L * per_step, "heads": det + heads,
            "total": stem + n_layers * L * per_step + det + heads}


use_gpu = torch.cuda.is_available() and not a.cpu_only
res = {"host": socket.gethostname(), "cpu": cpu_name(), "env": env_info(),
       "torch_threads_default": torch.get_num_threads(),
       "analytic_macs": {"base": analytic_macs(4), "mamba_l2": analytic_macs(2)}, "models": {}}
x = torch.randn(1, 3, 400)
for v in ("base", "lstm", "transformer", "mamba_l2"):
    entry = {}
    is_mamba = v in ("base", "mamba_l2")
    if use_gpu:
        m = build_model(v, backend="mamba_ssm" if is_mamba else "reference").eval()
        entry["params"] = sum(p.numel() for p in m.parameters())
        mg, xg, xb = m.cuda(), x.cuda(), torch.randn(64, 3, 400, device="cuda")
        with torch.no_grad():
            entry["gpu_b1"] = timeit(lambda: mg(xg), a.runs, torch.cuda.synchronize)
            t = timeit(lambda: mg(xb), max(50, a.runs // 10), torch.cuda.synchronize)
        entry["gpu_b64"] = t
        entry["gpu_windows_per_s_b64"] = 64 / (t["median_ms"] / 1e3)
    m = build_model(v, backend="reference").eval()
    entry["params"] = sum(p.numel() for p in m.parameters())
    if not is_mamba:
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(m, (3, 400), as_strings=False, print_per_layer_stat=False, verbose=False)
            entry["ptflops_macs"] = macs
        except Exception as e:
            entry["ptflops_macs"] = f"n/a ({type(e).__name__})"
    if is_mamba:  # pure-PyTorch reference scan, traced
        with torch.no_grad():
            m = torch.jit.trace(m, x, check_trace=False)
    for th in a.threads:
        torch.set_num_threads(th)
        with torch.no_grad():
            entry[f"cpu_torch_t{th}"] = timeit(lambda: m(x), a.runs if not is_mamba else max(50, a.runs // 5))
    if v == "base":
        try:
            import onnxruntime as ort
            path = os.path.join(tempfile.mkdtemp(), "pwavemamba.onnx")
            torch.onnx.export(build_model(v, backend="reference").eval(), x, path, opset_version=17)
            e = {"ort": ort.__version__}
            for th in a.threads:
                so = ort.SessionOptions()
                so.intra_op_num_threads = th
                s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
                name = s.get_inputs()[0].name
                e[f"cpu_t{th}"] = timeit(lambda: s.run(None, {name: x.numpy()}), a.runs)
            entry["onnxruntime"] = e
        except Exception as ex:
            entry["onnxruntime"] = f"n/a ({type(ex).__name__})"
    res["models"][v] = entry

out = os.path.join(a.out, "bench", f"{res['host']}{'_cpu' if a.cpu_only else ''}.json")
save_json(res, out)
print(json.dumps(res, indent=1, default=str)[:4000])
print("saved", out)
