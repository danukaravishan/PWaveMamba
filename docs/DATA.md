# Data

The waveforms are not part of this repository. They come from the GeoNet programme of New Zealand
(earthquake catalog doi:10.21420/0S8P-TZ38, waveform archive doi:10.21420/G19Y-9D40), licensed under
CC BY 3.0 NZ. All scripts expect the files below in `data/` (or pass `--data-dir`).

```
data/
  waveforms.hdf5            90 s three-component strong-motion records
  metadata.csv              one row per record, with the phase picks
  segments/                 written by scripts/extract_segments.py
  christchurch/
    christchurch.hdf5       records of the 22 Feb 2011 Mw 6.2 Christchurch earthquake
    metadata.csv
```

## Records used

GeoNet accelerometer records (HN channels at 200 Hz, BN channels at 50 Hz) from January 2013 to April 2025
of events with catalog magnitude M >= 3.0, within 100 km epicentral distance (150 km for M > 5.0),
corrected for instrument response to acceleration in m/s^2.
Each record is 90 s long (30 s before and 60 s after the P arrival); records missing a component were
discarded. 89,269 records passed all checks (Table I of the paper).

## `waveforms.hdf5`

Group `data`, one dataset per record named `<GeoNet event id>_<station>` (for example `2016p858000_KIKS`),
array of shape `(3, N)` in m/s^2. **Channel order:** the order of the `channels` attribute, which for every
record of the data set is (E or 1, N or 2, Z), i.e. the vertical component is the last row. The models were
trained with this order, so new data must be arranged the same way.

Dataset attributes read by the code:

| attribute | used for |
|---|---|
| `sampling_rate` | resampling to 100 Hz |
| `channels` | comma-separated channel codes, e.g. `HNE,HNN,HNZ` (identifies the horizontal components for the RotD50 PGA) |
| `magnitude` | magnitude target of the P window (copied into the segment file) |
| `p_arrival_sample` | must be present (any value); the pick actually used comes from `metadata.csv` |

All other attributes are copied unchanged into the segment file.

## `metadata.csv`

One row per record, indexed by `trace_name` (= the dataset name). Columns used:

| column | used by |
|---|---|
| `trace_name` | all |
| `magnitude` | split (stratification) |
| `new_p_arrival_sample` | P pick (sample index at the record's original sampling rate); records without it are skipped |
| `new_s_arrival_sample` | S pick (optional; no S window without it) |
| `epicentral_distance` (km), `event_depth` (km) | iasp91 S-P time for the PGA window |
| `station`, `magnitude_type` | copied into the per-window prediction tables |

The picks were made with PhaseNet and EQTransformer (SeisBench) and screened against the GeoNet catalog
and the iasp91 S-P time, as described in Section IV of the paper:
P picks were accepted when the PhaseNet confidence exceeded 0.9 or the pick lay within 1 s of a GeoNet
catalog arrival; S picks from either picker were accepted above confidence 0.8, and between 0.5 and 0.8
only within +-2 s of the iasp91 prediction.

## Windows (`scripts/extract_segments.py`)

From each record the script writes one P window, one S window and two noise windows into
`data/segments/segment_<window id>_<length>s.hdf5` (see the script header for the details and
`pwavemamba/config.py` for the window variants). The intensity target is
`ln(100 * RotD50 PGA [m/s^2]) + 5`, i.e. ln(PGA in cm/s^2) + 5; the MMI 5 threshold 42.3 cm/s^2 is 8.745 in
these units. The extraction is seeded (numpy seed 42): the same input files give the same windows.

## Split

`splits/event_split.csv` assigns every GeoNet event id to `train`, `validation` or `test` (event-grouped,
magnitude-stratified 80/10/10). It is computed by `scripts/make_split.py` from `metadata.csv` alone, without
random numbers, and is used by training and evaluation for every window length.

## Christchurch case study

`data/christchurch/christchurch.hdf5` holds the 106 strong-motion records of South Island stations for the
2011 Christchurch earthquake in the same format as `waveforms.hdf5`. `data/christchurch/metadata.csv` has one
row per record with `trace_name`, `station`, `epicentral_distance`, `sampling_rate`, `trace_starttime`, the P
and S arrivals (`phasenet_p_arrival_sample` / `p_arrival_sample` or the corresponding `*_time` columns, likewise
for S) and the reference shaking `pga_rotd50` (RotD50 PGA of the whole record, m/s^2).
