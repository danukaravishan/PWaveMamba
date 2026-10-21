"""Reading windows of one split straight from a segment file (no copies on disk)."""
import csv

import h5py
import numpy as np

from .preprocessing import segment_preprocess_data


def trace_of(window_id):
    """Trace name (<GeoNet event id>_<station>) of a window id ('noise_', 'noise_1_' and 's_' prefixes)."""
    for pre in ("noise_1_", "noise_", "s_"):
        if window_id.startswith(pre):
            return window_id[len(pre):]
    return window_id


def event_of(window_id):
    return trace_of(window_id).split("_")[0]


def read_split_csv(path):
    """{event_id: 'train' | 'validation' | 'test'} from splits/event_split.csv."""
    with open(path) as f:
        return {r["event_id"]: r["split"] for r in csv.DictReader(f)}


def load_split(split, segment_file, split_csv, window_samples, verbose=True):
    """All usable windows of one split.

    Windows with missing or NaN labels, NaN data or an invalid p_arrival_sample are skipped.
    Returns X (N, 3, L) float32 (preprocessed), y_mag, y_int (float32), y_det (N, L) int8,
    y_type (int64; 0 noise, 1 P, 2 S) and the window ids.
    """
    ev_split = read_split_csv(split_csv)
    L = window_samples
    with h5py.File(segment_file, "r") as f:
        g = f["positive_samples_p"]
        keys = [k for k in g.keys() if ev_split.get(event_of(k)) == split]
        X = np.empty((len(keys), 3, L), np.float32)
        y_det = np.empty((len(keys), L), np.int8)
        y_mag = np.empty(len(keys), np.float32)
        y_int = np.empty(len(keys), np.float32)
        y_type = np.empty(len(keys), np.int64)
        ids, n = [], 0
        for k in keys:
            d = g[k]
            at = d.attrs
            need = [at.get(a) for a in ("intensity_label", "magnitude_label", "time_mask", "detection_ts",
                                        "magnitude", "event_type", "pga")]
            if any(v is None for v in need):
                continue
            il, ml, tm, det, m, et, pga = need
            pa = at.get("p_arrival_sample")
            try:
                pa = float(pa)
            except (TypeError, ValueError):
                pa = None
            if pa is None or np.isnan(pa) or np.isnan(m) or np.isnan(et) or np.isnan(pga) \
                    or np.isnan(il).any() or np.isnan(ml).any() or np.isnan(tm).any() or np.isnan(det).any():
                continue
            x = segment_preprocess_data(np.array(d))
            if np.isnan(x).any() or x.shape != (3, L):
                continue
            X[n] = x
            y_det[n] = det
            y_mag[n] = m
            y_int[n] = pga
            y_type[n] = int(et)
            ids.append(k)
            n += 1
    if verbose:
        print(f"[load_split] {split}: {n} windows (of {len(keys)} in split) from {segment_file}")
    return X[:n], y_mag[:n], y_int[:n], y_det[:n], y_type[:n], np.array(ids)
