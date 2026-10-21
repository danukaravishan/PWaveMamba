# PWaveMamba

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23522642.svg)](https://doi.org/10.5281/zenodo.23522642)

Code, trained weights, data split and evaluation outputs for

> D. Ravishan, R. Prasanna, P. Herath and E. E. H. Doyle, "PWaveMamba: A Multi-Task Selective State-Space
> Model for Single-Station Earthquake Early Warning From the First Seconds of P-Wave Data,"
> submitted to *IEEE Transactions on Geoscience and Remote Sensing*.

PWaveMamba reads one 4 s window (2 s before and 2 s after the expected P arrival, 100 Hz, three components)
of strong-motion acceleration and returns, in one forward pass, per-sample probabilities of
{noise, P, S}, the expected peak ground acceleration (PGA) at the site and the event magnitude.
A causal first-exceedance rule turns the stream of window outputs into an onsite alert.

| | |
|---|---|
| Parameters | 238,561 |
| Backbone | convolutional stem + 4 selective SSM (Mamba) layers, d_model 52 |
| Input | 3 x 400 samples (4 s at 100 Hz), mu-law compressed |
| Outputs | detection logits (400 x 3), ln(PGA [cm/s^2]) + 5, magnitude |
| Training data | 89,269 GeoNet strong-motion records (2013-2025), event-disjoint split |

## Repository layout

```
pwavemamba/            the model and everything needed to run it
  model.py             PWaveMamba network
  mamba_ref.py         pure-PyTorch Mamba block (CPU / no-CUDA fallback, same parameters as mamba_ssm)
  variants.py          ablation variants (LSTM, Transformer, 2 layers, no P-conditioning, single task)
  preprocessing.py     input transform and band-pass used for noise windows
  data.py, losses.py, decision.py, config.py, utils.py
scripts/
  extract_segments.py  90 s records -> P, S and noise windows (+ labels)
  make_split.py        event-grouped, magnitude-stratified 80/10/10 split
  train.py             training recipe of the paper
  evaluate.py          threshold tuning on validation, test metrics, per-window predictions
  christchurch_eval.py 2011 Christchurch streaming case study and the Pd reference rule
  bench_compute.py     parameters, MACs, GPU/CPU latency
paper/
  christchurch_rules.py  first-exceedance alert rule, F1 vs. deadline (Fig. 3, Table III)
  main_tables.py         prints Tables I, III, IV, V from the saved outputs
  si_tables.py           Supplementary tables and figures
checkpoints/           14 trained models (about 1 MB each), see below
splits/                event_split.csv: GeoNet event id -> train / validation / test
results/               the outputs behind every number in the paper (see results/README.md)
docs/DATA.md           how to obtain and prepare the data
tests/smoke_test.py    end-to-end test on synthetic data (CPU)
run_all.sh             full reproduction on a GPU machine
```

## Installation

Training and the GPU timings use the CUDA kernels of `mamba-ssm` (Linux, NVIDIA GPU):

```bash
conda env create -f environment.yml
conda activate pwavemamba
pip install mamba-ssm==1.2.0.post1 --no-build-isolation     # needs a matching CUDA toolkit
```

Inference also runs without a GPU and without `mamba-ssm`: the model then uses
`pwavemamba/mamba_ref.py`, a step-by-step PyTorch implementation of the same selective scan with the same
parameter names. For CPU-only use, `pip install -r requirements.txt` is enough. On windows whose inputs are
identical, the reference implementation agrees with the CUDA-kernel outputs of the released models to about
1e-3 (float32 rounding).

## Quick start: run a released model

```python
import json
import numpy as np
from pwavemamba import build_model, load_checkpoint, segment_preprocess_data, predict, topk_conf, classify

model = build_model("base")                       # backend chosen automatically (mamba_ssm on GPU, else reference)
load_checkpoint(model, "checkpoints/base_s42.pt")

acc = np.random.randn(3, 400) * 1e-3  # one window: rows E (or 1), N (or 2), Z; 400 samples at 100 Hz; m/s^2
x = segment_preprocess_data(acc)[None].astype(np.float32)
mag, intensity, probs = predict(model, x, device="cpu")
cp, cs = topk_conf(probs)                          # top-60 mean P and S probability
t_p, t_s = json.load(open("results/eval/base_s42.json"))["test"]["thresholds_P_S"]   # validation-tuned
window_class = classify(cp, cs, t_p, t_s)          # 0 noise, 1 P, 2 S
pga_cms2 = np.exp(intensity - 5.0)                 # predicted RotD50 PGA in cm/s^2
```

The validation-tuned window thresholds of every model are stored in `results/eval/<model>.json`
under `test.thresholds_P_S`. For streaming use, slide the window with a 0.1 s hop and apply the
first-exceedance rule of the paper (see `paper/christchurch_rules.py`).

## Released checkpoints

| File | Model | Paper |
|---|---|---|
| `base_s42.pt`, `base_s7.pt`, `base_s123.pt` | PWaveMamba, three seeds | main results, Table III |
| `mamba_l2_s42.pt` | 2 Mamba layers | Table III |
| `no_pcond_s42.pt` | without P-conditioned pooling | Table III |
| `det_only_s42.pt`, `int_only_s42.pt`, `mag_only_s42.pt` | single-task models | Table III |
| `lstm_s42.pt`, `transformer_s42.pt` | equal-size LSTM and causal Transformer backbones | Tables III, IV |
| `base_s42_ID2.pt`, `_ID10`, `_ID13`, `_ID14` | 2 s before P and 1, 3, 4, 5 s after P | Table V |

Each file is a dict with `model_state_dict`, `variant`, `seed`, `best_epoch` and `epochs_run`.
Load it with `build_model(variant, window_id)` and `load_checkpoint(...)`; window ids are listed in
`pwavemamba/config.py`.

## Reproducing the paper

1. Prepare `data/` as described in [docs/DATA.md](docs/DATA.md).
2. On a GPU machine with `mamba-ssm`: `JOBS=3 bash run_all.sh`. This extracts the windows, checks that the
   released split is reproduced, trains all 14 models, evaluates them, runs the Christchurch case study and
   the benchmark, and prints the tables.
3. Individual steps:

```bash
python scripts/extract_segments.py --window ID6
python scripts/train.py --variant base --seed 42
python scripts/evaluate.py --ckpt outputs/models/base_s42.pt
python scripts/christchurch_eval.py --ckpt outputs/models/base_s42.pt --eval-json outputs/eval/base_s42.json
python paper/christchurch_rules.py outputs
python paper/main_tables.py outputs
```

Without the data, the tables can be regenerated from the released outputs:

```bash
python paper/christchurch_rules.py results
python paper/main_tables.py results
```

Training on a GPU is not bit-for-bit deterministic, so a re-run reproduces the released numbers up to the
seed-to-seed spread reported in the paper.

To check an installation without the data: `python tests/smoke_test.py`.

## Data

The waveforms and the earthquake catalog are from the GeoNet programme of New Zealand and are not
redistributed in this repository. GeoNet data are licensed under CC BY 3.0 NZ; we acknowledge the New Zealand
GeoNet programme and its sponsors NHC, Earth Sciences NZ, LINZ, NEMA and MBIE for providing the data used in
this study. See [docs/DATA.md](docs/DATA.md).

## Citation

If you use this code or the trained models, please cite the paper (reference will be updated on
publication) and the archived release:

> D. Ravishan, R. Prasanna, P. Herath and E. E. H. Doyle, "PWaveMamba: code, trained models and evaluation
> outputs," version 1.0.0, Zenodo, 2026, doi: 10.5281/zenodo.23522642.

See also `CITATION.cff`.

## License

Code and trained weights: MIT License (see `LICENSE`).
