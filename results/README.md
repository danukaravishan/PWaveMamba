# Released outputs

Every number in the paper is computed from the files in this folder. They are the raw outputs of the
scripts in `scripts/`, run on an NVIDIA H100 NVL with PyTorch 2.2.2 and mamba-ssm 1.2.0.post1, with three
changes for publication: checkpoint paths were reduced to file names, an auxiliary evaluation at fixed
thresholds was removed from the `eval/*.json` files, and the per-window CSV files were gzip-compressed.

| Folder | Content | Paper |
|---|---|---|
| `eval/<model>.json` | tuned thresholds, confusion matrices, PR curves, intensity and magnitude metrics (overall and by magnitude bin) on the test set | Tables III, V; Fig. 2 |
| `eval/<model>_test_predictions.csv.gz` | one row per test window: true and predicted class, P/S confidence, true and predicted intensity and magnitude, distance, station | Fig. 2; SI |
| `eval/record_split_ref.json` | an earlier model trained and tested on a record-wise random split (checkpoint not released) | SI, effect of the split |
| `train/<model>.json` | training and validation loss per epoch, best epoch, run time, software versions | SI |
| `christchurch/<model>_windows.csv.gz` | every 4 s window (0.1 s hop) of the 2011 Christchurch records: class, P/S confidence, intensity, magnitude | Section VII |
| `christchurch/<model>_stations.csv`, `_summary.json` | per-record arrivals, reference PGA, Pd, hold-rule alerts | Section VII |
| `bench/*.json` | parameters, MACs, GPU and CPU latency | Table IV |
| `optuna/*.db` | Optuna studies of the hyperparameter search (SQLite) | Table II; SI |
| `paper/` | `christchurch_rules.json`, `christchurch_tradeoff.csv`: first-exceedance rule (from `paper/christchurch_rules.py`) | Fig. 3; Table III |

Model names: `<variant>_s<seed>[_<window id>]`, see the main README.
Intensity values are ln(PGA [cm/s^2]) + 5.
