"""Build the training windows (segment file) from the 90 s GeoNet records.

    python scripts/extract_segments.py --window ID6        # -> data/segments/segment_ID6_4s.hdf5

For every record of data/waveforms.hdf5 (group "data") that has a P pick in data/metadata.csv
(column new_p_arrival_sample) this writes, into the group "positive_samples_p":
  * <trace>           P window: `before` s before and `after` s after the P pick, with a uniform random
                      shift of +-0.5 s; labels: detection mask (1 from the P arrival), intensity target
                      ln(100 * RotD50 PGA [m/s^2]) + 5 and the catalog magnitude.
  * s_<trace>         S window around the S pick (column new_s_arrival_sample), shift +-0.25 s, label 2.
  * noise_<trace>     raw pre-event noise ending >= 2 s before P;  noise_1_<trace>: the same after
                      demean/detrend/1-20 Hz band-pass. Noise windows are augmented with white Gaussian noise
                      (p = 1/4) and impulsive "footstep" bursts (p = 1/8).
Records are resampled to 100 Hz. The RotD50 PGA is computed from the horizontal components starting at
the iasp91 S-P time after P (at most 3 s after P). The random numbers come from numpy with seed 42, so a
run over the same input files gives the same windows. Windows are stored as raw acceleration; the network
transform (mu-law) is applied at load time.

Needs numpy, scipy, pandas, h5py and obspy (TauP). Window variants: see pwavemamba/config.py.
"""
import argparse
import os
import random

import h5py
import numpy as np
import pandas as pd
import scipy.signal
from obspy.geodetics import kilometers2degrees
from obspy.taup import TauPyModel

import _common
from pwavemamba.config import SAMPLING_RATE, SEED, WINDOWS, segment_filename
from pwavemamba.preprocessing import pre_proc_data

# Global TauPyModel instance (loaded lazily to avoid startup cost)
_TAUP_MODEL = None


class _Variation:
    def __init__(self, name):
        self.name, self.value = name, list(WINDOWS[name])


class Cfg:
    """The configuration fields used by extract_data()."""

    def __init__(self, data_dir, window_id, waveforms=None, metadata=None):
        self.SEED = SEED
        self.BASE_SAMPLING_RATE = SAMPLING_RATE
        self.DATASET_VARIATION = _Variation(window_id)
        self.ORIGINAL_DB_FILE = waveforms or os.path.join(data_dir, "waveforms.hdf5")
        self.METADATA_PATH = metadata or os.path.join(data_dir, "metadata.csv")
        self.DATA_EXTRACTED_FILE = os.path.join(data_dir, "segments", segment_filename(window_id))


def _get_taup_model():
    """Lazily initialize the IASP91 TauPyModel."""
    global _TAUP_MODEL
    if _TAUP_MODEL is None:
        _TAUP_MODEL = TauPyModel(model="iasp91")
    return _TAUP_MODEL


def compute_rotd50_pga(a1: np.ndarray, a2: np.ndarray, angle_step: int = 1) -> float:
    """Compute RotD50 PGA from two horizontal acceleration components.
    
    RotD50 is the median of the peak ground accelerations computed
    for rotations from 0 to 180 degrees.
    """
    if a1.size == 0 or a2.size == 0:
        return float("nan")

    angles = np.deg2rad(np.arange(0, 180, angle_step, dtype=float))
    pga_vals = []
    for th in angles:
        c, s = np.cos(th), np.sin(th)
        r = a1 * c + a2 * s
        pga = np.max(np.abs(r))
        pga_vals.append(pga)
    return float(np.nanmedian(pga_vals)) if pga_vals else float("nan")


def estimate_s_arrival_offset(epicentral_distance_km: float, depth_km: float) -> float:
    """Estimate the S-P travel time difference using IASP91 model.
    
    Returns the estimated time (in seconds) between P and S wave arrivals.
    Returns None if calculation fails.
    """
    try:
        model = _get_taup_model()
        dist_deg = kilometers2degrees(epicentral_distance_km)
        
        arrivals = model.get_travel_times(
            source_depth_in_km=depth_km,
            distance_in_degree=dist_deg,
            phase_list=["p", "P", "s", "S"]
        )
        
        p_travel_time = None
        s_travel_time = None
        
        for arrival in arrivals:
            phase_name = arrival.name.upper()
            if p_travel_time is None and phase_name.startswith("P"):
                p_travel_time = arrival.time
            if s_travel_time is None and phase_name.startswith("S"):
                s_travel_time = arrival.time
            if p_travel_time is not None and s_travel_time is not None:
                break
        
        if p_travel_time is not None and s_travel_time is not None:
            return s_travel_time - p_travel_time
        return None
    except Exception:
        return None


def calculate_pga_from_waveform(
    data: np.ndarray,
    channels: list,
    p_arrival_sample: int,
    sampling_rate: int,
    epicentral_distance_km: float = None,
    depth_km: float = None,
    default_window_sec: float = 3.0
) -> float:
    """Calculate RotD50 PGA from waveform data using dynamic S-wave window.
    
    The window start is determined as follows:
    1. Estimate S-wave arrival time using IASP91 model from epicentral distance and depth
    2. If estimated S-P time is between 0 and 3 seconds, use it as the window start
    3. Otherwise, default to 3 seconds after P arrival
    
    Parameters:
    -----------
    data : np.ndarray
        Waveform data with shape (n_channels, n_samples)
    channels : list
        List of channel names (e.g., ['HNE', 'HNN', 'HNZ'])
    p_arrival_sample : int
        Sample index of P-wave arrival
    sampling_rate : int
        Sampling rate in Hz
    epicentral_distance_km : float, optional
        Epicentral distance in kilometers (for S-wave estimation)
    depth_km : float, optional
        Event depth in kilometers (for S-wave estimation)
    default_window_sec : float
        Default time window after P arrival (default 3.0 seconds)
    
    Returns:
    --------
    float : RotD50 PGA value, or NaN if calculation fails
    """
    if data.ndim != 2 or data.shape[0] < 2:
        return float("nan")
    
    # Determine window start offset based on S-wave estimation
    window_offset_sec = default_window_sec
    
    if epicentral_distance_km is not None and depth_km is not None:
        s_p_offset = estimate_s_arrival_offset(epicentral_distance_km, depth_km)
        if s_p_offset is not None and s_p_offset > 0 and s_p_offset < default_window_sec:
            window_offset_sec = s_p_offset
    
    # Calculate start index
    start_index = p_arrival_sample + int(window_offset_sec * sampling_rate)
    
    if start_index >= data.shape[1]:
        return float("nan")
    
    # Identify horizontal channels
    channel_indexes = []
    for idx, channel in enumerate(channels):
        ch_upper = channel.strip().upper()
        if ch_upper.endswith("Z"):
            continue
        channel_indexes.append(idx)
    
    if len(channel_indexes) < 2:
        return float("nan")
    
    # Extract horizontal components from start_index onwards
    a1 = data[channel_indexes[0], start_index:]
    a2 = data[channel_indexes[1], start_index:]
    
    return compute_rotd50_pga(a1, a2)


def set_global_seed(seed: int) -> None:
    """Set global seed for reproducibility across numpy/random."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"[Seed] Global seed set to {seed}")


def extract_wave_window_two_sided(data, wave_index, samples_before, samples_after, pad_value=0.0):
    """
    Extract a window around `wave_index` using explicit samples before and after the pick.

    Parameters
    - data: np.ndarray with shape (n_channels, n_samples)
    - wave_index: int index of the pick (sample index)
    - samples_before: number of samples to include before wave_index
    - samples_after: number of samples to include after wave_index
    - pad_value: value used to pad when the requested window extends beyond data edges

    Returns
    - window: np.ndarray of shape (n_channels, samples_before + samples_after)

    The function always returns a fixed-length window. If the requested start or end
    exceeds the available data, the returned window is padded with `pad_value`.
    """
    n_channels, n_samples = data.shape

    start = int(wave_index - samples_before)
    end = int(wave_index + samples_after)

    # Desired window length
    desired_len = samples_before + samples_after

    # Compute overlap with actual data
    data_start = max(0, start)
    data_end = min(n_samples, end)

    # Initialize output with pad_value
    window = np.full((n_channels, desired_len), pad_value, dtype=data.dtype)

    # Determine where to place the available data in the window
    insert_start = data_start - start if start < data_start else 0
    insert_end = insert_start + (data_end - data_start)

    if data_end > data_start:
        window[:, insert_start:insert_end] = data[:, data_start:data_end]

    return window


def resample_data(data, original_rate, target_rate):
    """
    Resamples the data to the target_rate using FFT-based resampling (scipy.signal.resample).
    This handles both upsampling and downsampling for non-integer factors correctly.
    """
    if target_rate == original_rate:
        return data
    
    # Calculate new number of samples
    num_samples = int(data.shape[1] * target_rate / original_rate)
    return scipy.signal.resample(data, num_samples, axis=1)


def generate_intensity_label(length, p_arrival_index, pga_value):
    label = np.zeros(length)
    tau_I = 2
    t_values = np.arange(length)
    
    mask = t_values >= p_arrival_index
    if np.any(mask):
        t_diff = t_values[mask] - p_arrival_index
        label[mask] = pga_value * (1 - np.exp(-t_diff / tau_I))
            
    return label


def generate_magnitude_label(length, p_arrival_index, magnitude_value):
    label = np.zeros(length)
    tau_I = 4
    t_values = np.arange(length)
    
    #magnitude_value = 100 * magnitude_value
    mask = t_values >= p_arrival_index
    if np.any(mask):
        t_diff = t_values[mask] - p_arrival_index
        label[mask] = magnitude_value * (1 - np.exp(-t_diff / tau_I))
            
    return label


def generate_time_mask(length, p_arrival_index, value=1.0):
    mask = np.zeros(length, dtype=np.float32)
    start = max(0, int(p_arrival_index))
    if start < length:
        mask[start:] = value
    return mask


def add_gaussian_noise(segment, gaussian_std_scale=0.05):
    """
    Add white Gaussian noise to a waveform segment.

    Parameters
    - segment: np.ndarray (n_channels, n_samples)
    - gaussian_std_scale: multiplier on segment std for Gaussian noise
    """
    augmented = segment.copy()
    seg_std = float(np.std(augmented))
    if seg_std <= 0:
        seg_std = 1e-6

    gaussian_std = gaussian_std_scale * seg_std
    gaussian_noise = np.random.normal(loc=0.0, scale=gaussian_std, size=augmented.shape)
    augmented += gaussian_noise
    return augmented


def add_impulse_noise(segment, sampling_rate, impulse_scale=20.0, footstep_duration_range=(0.05, 0.18)):
    """
    Add a footstep-like impulse to a waveform segment.

    Parameters
    - segment: np.ndarray (n_channels, n_samples)
    - sampling_rate: sampling rate in Hz
    - impulse_scale: multiplier on segment std for impulse amplitude
    - footstep_duration_range: duration range in seconds for the thump
    """

    impulse_scale = np.random.randint(5,20)
    augmented = segment.copy()
    seg_std = float(np.std(augmented))
    if seg_std <= 0:
        seg_std = 1e-6

    n_channels, n_samples = augmented.shape

    dur_sec = np.random.uniform(footstep_duration_range[0], footstep_duration_range[1])
    impulse_len = max(3, int(dur_sec * sampling_rate))
    if impulse_len >= n_samples:
        impulse_len = max(3, n_samples - 1)

    start_idx = np.random.randint(0, n_samples - impulse_len)
    t = np.linspace(0.0, 1.0, impulse_len)

    # Footstep-like thump: raised-cosine with exponential decay
    envelope = (np.sin(np.pi * t) ** 2) * np.exp(-3.5 * t)
    thump = envelope * (impulse_scale * seg_std)

    # Apply to all channels with slight scaling differences
    channel_scales = 0.7 + 0.6 * np.random.rand(n_channels)
    for ch in range(n_channels):
        augmented[ch, start_idx:start_idx + impulse_len] += thump * channel_scales[ch]

    return augmented


def extract_data(cfg):
    set_global_seed(cfg.SEED)

    print(f"Extracting {cfg.DATA_EXTRACTED_FILE} from {cfg.ORIGINAL_DB_FILE}")

    hdf5_file = h5py.File(cfg.ORIGINAL_DB_FILE, 'r')

    if os.path.isfile(cfg.DATA_EXTRACTED_FILE):
        os.remove(cfg.DATA_EXTRACTED_FILE)

    with h5py.File(cfg.DATA_EXTRACTED_FILE, 'a') as hdf:
        
        # Create database groups
        if 'positive_samples_p' not in hdf:
            positive_group_p = hdf.create_group('positive_samples_p')
        else:
            positive_group_p = hdf['positive_samples_p']


        count = 0
        non_p_arrival_count = 0

        # Load metadata DataFrame once before the loop
        
        metadata_df = pd.read_csv(cfg.METADATA_PATH)
        metadata_df.set_index('trace_name', inplace=True)

        if "new_p_arrival_sample" not in metadata_df.columns:
            raise KeyError("Metadata missing 'new_p_arrival_sample' column")
        
        index = 0

        for event_id in hdf5_file["data"].keys():  # Directly iterate over the keys in the HDF5 file

            index += 1
            if index % 500 == 0:
                print(f"Processed {index} records...")
                
            # Temporarily limit the number of records for testing

            dataset = hdf5_file["data"].get(event_id)
            data = np.array(dataset)

            # Read pick channel from metadata
            if event_id not in metadata_df.index:
                print(f"Skipping event {event_id}: not found in metadata.")
                continue

            # pick_channel = str(metadata_df.loc[event_id, 'pick_channel']).strip()
            # if pick_channel == "":
            #     print(f"Skipping event {event_id}: empty pick_channel")
            #     continue

            p_arrival_sample = metadata_df.loc[event_id, "new_p_arrival_sample"]

            if p_arrival_sample is None or p_arrival_sample == "" or pd.isna(p_arrival_sample):
                non_p_arrival_count +=1
                continue

            try:
                p_arrival_sample = float(p_arrival_sample)
                
            except Exception:
                print(f"Skipping event {event_id}: non-numeric new_p_arrival_sample ({p_arrival_sample})")
                non_p_arrival_count +=1
                continue

            if p_arrival_sample > len(data[0]):
                print(f"Skipping event {event_id}: p_arrival_sample {p_arrival_sample} exceeds data length {len(data[0])}")
                non_p_arrival_count +=1
                continue

            p_arrival_index = int(p_arrival_sample)
            p_arrival_index_before_resalpming = p_arrival_index

            ## S-wave Prep
            has_s_wave = False
            s_arrival_index = 0
            if "new_s_arrival_sample" in metadata_df.columns:
                s_sample = metadata_df.loc[event_id, "new_s_arrival_sample"]
                if s_sample is not None and s_sample != "" and not pd.isna(s_sample):
                    try:
                        s_arrival_index = int(float(s_sample))
                        has_s_wave = True
                    except:
                        pass
            
            s_arrival_index_before_resalpming = int(s_arrival_index)            
            sampling_rate = int(dataset.attrs["sampling_rate"])

            #data_pre_proc = pre_proc_data(data, sampling_rate=sampling_rate)
            data_pre_proc = data

            if sampling_rate != cfg.BASE_SAMPLING_RATE:
                original_length = data_pre_proc.shape[1]
                data_pre_proc = resample_data(data_pre_proc, sampling_rate, cfg.BASE_SAMPLING_RATE)
                new_length = data_pre_proc.shape[1]
                p_arrival_index = int(p_arrival_index * new_length / original_length)
                if has_s_wave:
                    s_arrival_index = int(s_arrival_index * new_length / original_length)
                sampling_rate = cfg.BASE_SAMPLING_RATE
                
            if data_pre_proc.shape[1] <= p_arrival_index:
                continue
            
            if p_arrival_index < 0:
                continue

            ## Give temporal shift to the data
            SHIFT_RANGE_SEC =  0.5 # Max shift in seconds (±0.5 sec)
            
            shift_samples = int(SHIFT_RANGE_SEC * sampling_rate)
            random_shifts_p = np.random.randint(-shift_samples, shift_samples + 1)

            # Modify P and S indices
            new_p_indices = np.clip(p_arrival_index + random_shifts_p, 0, len(data_pre_proc[0]) - 1)

            # Extract wave data
            samples_before = int(cfg.DATASET_VARIATION.value[0]*sampling_rate)
            samples_after = int(cfg.DATASET_VARIATION.value[1]*sampling_rate)
            window_size = samples_before + samples_after
            
            #p_data  = extract_wave_window(data_pre_proc, new_p_indices, window_size)
            p_data  = extract_wave_window_two_sided(data_pre_proc, new_p_indices, samples_before, samples_after)
            
            if len(p_data[0]) != window_size:
                print("Wrong data  ====== : "+event_id)
                continue
            
            # Get channel information for PGA calculation
            attr_channels = dataset.attrs.get("channels", "")
            channels = [ch.strip().upper() for ch in attr_channels.split(",") if ch.strip()]
            
            # Get epicentral distance and depth for S-wave estimation
            epicentral_distance_km = None
            depth_km = None
            if "epicentral_distance" in metadata_df.columns:
                epi_dist = metadata_df.loc[event_id, "epicentral_distance"]
                if not pd.isna(epi_dist):
                    epicentral_distance_km = float(epi_dist)
            if "event_depth" in metadata_df.columns:
                ev_depth = metadata_df.loc[event_id, "event_depth"]
                if not pd.isna(ev_depth):
                    depth_km = float(ev_depth)
            
            # Calculate PGA using the new dynamic S-wave window function
            pga = calculate_pga_from_waveform(
                data=data,
                channels=channels,
                p_arrival_sample=p_arrival_index_before_resalpming,
                sampling_rate=int(dataset.attrs["sampling_rate"]),
                epicentral_distance_km=epicentral_distance_km,
                depth_km=depth_km,
                default_window_sec=3.0
            )

            if np.isnan(pga):
                print(event_id + " has no PGA value, skipping...")
                continue

            # Better representation of pga is transformation to the log scale, withh GAL
            # Transform the pga. We need to re transform when converting back to other scales
            pga = np.log(100*float(pga))
            ## Shift to ensure positivity. Lowest PGA_rotd50 in metadata: 0.0002188384532928, log value = -3.822006570559144
            pga = pga + 5.0 

            ## Add data to each groups
            if event_id not in positive_group_p:
                positive_p_dataset = positive_group_p.create_dataset(event_id, data=p_data)
            else:
                print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")

            # Copy all original attributes
            for key, value in dataset.attrs.items():
                positive_group_p[event_id].attrs[key] = value

            # Set specific attributes
            magnitude = metadata_df.loc[event_id, 'magnitude']

            positive_group_p[event_id].attrs["pga"] = pga

            # Peak P-wave (Z channel) attributes from the extracted window
            z_channel = p_data[0]
            peak_p = float(np.max(np.abs(z_channel)))
            peak_p_log = np.log(100 * peak_p) + 5
            positive_group_p[event_id].attrs["peak_p"] = peak_p
            positive_group_p[event_id].attrs["peak_p_log"] = peak_p_log
            
            # The P-arrival in the new segment is located at: samples_before - random_shift
            p_arrival_rel = samples_before - random_shifts_p
            positive_group_p[event_id].attrs["p_arrival_sample"] = p_arrival_rel
            positive_group_p[event_id].attrs["p_arrival_original_index"] = p_arrival_index_before_resalpming

            intensity_label = generate_intensity_label(window_size, p_arrival_rel, pga)
            magnitude_label = generate_magnitude_label(window_size, p_arrival_rel, magnitude)
            time_mask = generate_time_mask(window_size, p_arrival_rel)
            detection_ts = generate_time_mask(window_size, p_arrival_rel, value=1.0)

            
            positive_group_p[event_id].attrs["intensity_label"] = intensity_label
            positive_group_p[event_id].attrs["magnitude_label"] = magnitude_label
            positive_group_p[event_id].attrs["time_mask"] = time_mask
            positive_group_p[event_id].attrs["event_type"] = 1
            positive_group_p[event_id].attrs["detection_ts"] = detection_ts
            

            # --- Noise Extraction ---
            # Extract noise anywhere from start up to 2 seconds before the P arrival.
            noise_pre_buffer = int(2.0 * sampling_rate)

            max_start = p_arrival_index - noise_pre_buffer - window_size

            valid_intervals = []
            # Interval: [0, max_start] (Inclusive)
            if max_start >= 0:
                valid_intervals.append((0, int(max_start)))
                
            total_points = sum(end - start + 1 for start, end in valid_intervals)
            
            if total_points > 0:
                rand_pt = np.random.randint(0, total_points)
                current = 0
                noise_start_index = -1
                for start, end in valid_intervals:
                    span = end - start + 1
                    if rand_pt < current + span:
                        noise_start_index = start + (rand_pt - current)
                        break
                    current += span

                noise_window = data_pre_proc[:, noise_start_index : noise_start_index + window_size]
                data_pre_proc_for_noise = pre_proc_data(data_pre_proc, sampling_rate=sampling_rate)
                
                if noise_window.shape[1] == window_size:
                    # Gaussian noise on binary random basis
                    apply_gaussian = np.random.randint(0, 4) == 1
                    if apply_gaussian:
                        noise_window = add_gaussian_noise(noise_window)

                    # Impulsive noise with probability 1/8 (footstep-like thump)
                    apply_impulse = np.random.randint(0, 8) == 0
                    if apply_impulse:
                        noise_window = add_impulse_noise(noise_window, sampling_rate)

                    noise_event_id = f"noise_{event_id}"
                    
                    if noise_event_id not in positive_group_p:
                        noise_ds = positive_group_p.create_dataset(noise_event_id, data=noise_window)
                        
                        # Copy original attributes
                        for key, value in dataset.attrs.items():
                            noise_ds.attrs[key] = value
                            
                        # Overwrite specific attrs for noise
                        noise_ds.attrs["pga"] = 0
                        noise_ds.attrs["magnitude"] = 0
                        noise_ds.attrs["p_arrival_sample"] = -1 # Dummy valid value for loader
                        noise_ds.attrs["p_arrival_original_index"] = -1
                        noise_ds.attrs["event_type"] = 0
                        
                        noise_ds.attrs["intensity_label"] = np.zeros(window_size)
                        noise_ds.attrs["magnitude_label"] = np.zeros(window_size)
                        noise_ds.attrs["time_mask"] = np.zeros(window_size)
                        noise_ds.attrs["detection_ts"] = np.zeros(window_size)

                # Extract another random noise segment from data_pre_proc_for_noise
                rand_pt_2 = np.random.randint(0, total_points)
                current_2 = 0
                noise_start_index_2 = -1
                for start, end in valid_intervals:
                    span = end - start + 1
                    if rand_pt_2 < current_2 + span:
                        noise_start_index_2 = start + (rand_pt_2 - current_2)
                        break
                    current_2 += span

                noise_window_2 = data_pre_proc_for_noise[:, noise_start_index_2 : noise_start_index_2 + window_size]

                if noise_window_2.shape[1] == window_size:
                    # Gaussian noise on binary random basis
                    apply_gaussian = np.random.randint(0, 4) == 1
                    if apply_gaussian:
                        noise_window_2 = add_gaussian_noise(noise_window_2)

                    # Impulsive noise with probability 1/8 (footstep-like thump)
                    apply_impulse = np.random.randint(0, 8) == 0
                    if apply_impulse:
                        noise_window_2 = add_impulse_noise(noise_window_2, sampling_rate)

                    noise_event_id_2 = f"noise_1_{event_id}"

                    if noise_event_id_2 not in positive_group_p:
                        noise_ds_2 = positive_group_p.create_dataset(noise_event_id_2, data=noise_window_2)

                        # Copy original attributes
                        for key, value in dataset.attrs.items():
                            noise_ds_2.attrs[key] = value

                        # Overwrite specific attrs for noise
                        noise_ds_2.attrs["pga"] = 0
                        noise_ds_2.attrs["magnitude"] = 0
                        noise_ds_2.attrs["p_arrival_sample"] = -1
                        noise_ds_2.attrs["p_arrival_original_index"] = -1
                        noise_ds_2.attrs["event_type"] = 0

                        noise_ds_2.attrs["intensity_label"] = np.zeros(window_size)
                        noise_ds_2.attrs["magnitude_label"] = np.zeros(window_size)
                        noise_ds_2.attrs["time_mask"] = np.zeros(window_size)
                        noise_ds_2.attrs["detection_ts"] = np.zeros(window_size)


            # --- S-wave Extraction ---
            if has_s_wave and s_arrival_index < data_pre_proc.shape[1] and s_arrival_index >= 0:
                
                # Use same shift range as P-wave
                SHIFT_RANGE_SEC = 0.25 
                shift_samples_s = int(SHIFT_RANGE_SEC * sampling_rate)
                random_shifts_s = np.random.randint(-shift_samples_s, shift_samples_s + 1)
                
                new_s_indices = np.clip(s_arrival_index + random_shifts_s, 0, len(data_pre_proc[0]) - 1)
                
                s_data = extract_wave_window_two_sided(data_pre_proc, new_s_indices, samples_before, samples_after)
                
                if len(s_data[0]) == window_size:
                    s_event_id = f"s_{event_id}"
                    
                    if s_event_id not in positive_group_p:
                        s_ds = positive_group_p.create_dataset(s_event_id, data=s_data)
                        
                        # Copy original attributes
                        for key, value in dataset.attrs.items():
                            s_ds.attrs[key] = value
                            
                        s_ds.attrs["pga"] = 0
                        s_ds.attrs["magnitude"] = 0
                        
                        s_arrival_rel = samples_before - random_shifts_s
                        s_ds.attrs["s_arrival_sample"] = s_arrival_rel 
                        s_ds.attrs["s_arrival_original_index"] = s_arrival_index_before_resalpming
                        s_ds.attrs["event_type"] = 2
                        
                        s_ds.attrs["intensity_label"] = np.zeros(window_size)
                        s_ds.attrs["magnitude_label"] = np.zeros(window_size)
                        s_ds.attrs["time_mask"] = np.zeros(window_size)
                        s_ds.attrs["detection_ts"] = generate_time_mask(window_size, s_arrival_rel, value=2.0)

            # if count > 1000:
            #     break
            # count += 1


    hdf5_file.close() 

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the window segment file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(_common.ROOT, "data"))
    ap.add_argument("--window", default="ID6", choices=list(WINDOWS))
    ap.add_argument("--waveforms", help="default: <data-dir>/waveforms.hdf5")
    ap.add_argument("--metadata", help="default: <data-dir>/metadata.csv")
    a = ap.parse_args()
    cfg = Cfg(a.data_dir, a.window, a.waveforms, a.metadata)
    os.makedirs(os.path.dirname(cfg.DATA_EXTRACTED_FILE), exist_ok=True)
    extract_data(cfg)
