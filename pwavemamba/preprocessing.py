"""Waveform preprocessing.

`segment_preprocess_data` is the only transform applied to a window before the network
(mu-law compression with mu = 100, then x100). `pre_proc_data` (demean, detrend, 1-20 Hz band-pass,
taper) is used only to build the filtered noise windows during segment extraction.
"""
import numpy as np
from scipy.signal import butter, detrend, filtfilt


def demean(signal):
    return signal - np.mean(signal)


def apply_detrend(signal):
    return detrend(signal)


def bandpass_filter(signal, lowcut=1, highcut=20.0, fs=100, order=4):
    """Zero-phase Butterworth band-pass, robust to short inputs."""
    if signal is None:
        return signal
    n = int(np.size(signal))
    if n <= 3:
        return signal
    nyquist = 0.5 * fs
    if nyquist <= 0:
        return signal
    low = max(1e-6, min(0.999999, float(lowcut) / nyquist))
    high = max(1e-6, min(0.999999, float(highcut) / nyquist))
    if not (0 < low < high < 1):
        return signal
    b, a = butter(order, [low, high], btype="band")
    padlen = max(0, min(3 * (max(len(a), len(b)) - 1), n - 1))
    try:
        return filtfilt(b, a, signal, padlen=padlen)
    except ValueError:
        return filtfilt(b, a, signal, method="gust")


def pre_proc_data(data, sampling_rate, pad_len=50, taper_len=20):
    """Demean, detrend, reflect-pad, 1-20 Hz band-pass and Hann taper, per channel. data: (C, N)."""
    processed = []
    for sig in data:
        x = apply_detrend(demean(sig))
        if pad_len and pad_len > 0 and x.size > pad_len:
            x = bandpass_filter(np.pad(x, pad_width=pad_len, mode="reflect"), fs=sampling_rate)[pad_len:-pad_len]
        else:
            x = bandpass_filter(x, fs=sampling_rate)
        if taper_len and taper_len > 0 and x.size > 2 * taper_len:
            taper = np.ones_like(x)
            window = np.hanning(2 * taper_len)
            taper[:taper_len] = window[:taper_len]
            taper[-taper_len:] = window[-taper_len:]
            x = x * taper
        processed.append(x)
    return np.array(processed)


def log_transform_data(data, mu=1000.0):
    """mu-law compression per channel for (C, N) or (B, C, N) arrays."""
    mu = float(mu)
    if mu <= 0:
        raise ValueError("mu must be > 0")
    denom = np.log1p(mu)
    f = lambda s: np.sign(s) * np.log1p(mu * np.abs(s)) / denom
    if data.ndim == 3:
        return np.array([np.array([f(sig[i, :]) for i in range(sig.shape[0])]) for sig in data])
    if data.ndim == 2:
        return np.array([f(data[i, :]) for i in range(data.shape[0])])
    raise ValueError("Input data must have 2 or 3 dimensions.")


def segment_preprocess_data(data):
    """Network input transform for one (3, L) window of acceleration in m/s^2."""
    return log_transform_data(data, mu=100.0) * 1e2
