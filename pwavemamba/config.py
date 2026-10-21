"""Fixed configuration of PWaveMamba as used in the paper.

All values are copied from the training code that produced the released checkpoints.
"""

SAMPLING_RATE = 100  # Hz, every record is resampled to this rate
SEED = 42            # seed of the segment extraction (random window shifts, noise augmentation)

# Window variants: (seconds before the phase pick, seconds after it).
# ID6 (2 s + 2 s = 4 s, L = 400) is the main model; the others are the window-length study.
WINDOWS = {
    "ID2": (2, 1),
    "ID6": (2, 2),
    "ID10": (2, 3),
    "ID13": (2, 4),
    "ID14": (2, 5),
}
DEFAULT_WINDOW = "ID6"


def window_seconds(window_id):
    before, after = WINDOWS[window_id]
    return before + after


def window_samples(window_id):
    return SAMPLING_RATE * window_seconds(window_id)


def segment_filename(window_id):
    """File name of the extracted segment file of one window variant (inside <data>/segments/)."""
    return f"segment_{window_id}_{window_seconds(window_id)}s.hdf5"


def tag_suffix(window_id):
    """Checkpoint / result name suffix: '' for the main window, '_ID10' etc. otherwise."""
    return "" if window_id == DEFAULT_WINDOW else f"_{window_id}"


# Network hyperparameters (Optuna-selected; Table II of the paper).
MODEL_HPARAMS = dict(
    d_model=52,
    n_layers=4,
    d_state=16,
    d_conv=16,
    expand=3,
    fc1_neurons=128,
    fc2_neurons=128,
    dropout=0.23678953816166126,
    mag_dropout=0.19136065535590244,
    int_dropout=0.2761299152853056,
    det_dropout=0.2074003587466754,
    stem_conv1_filters=32,
    stem_conv2_filters=32,
    stem_conv3_filters=48,
    stem_kernel_size1=3,
    stem_kernel_size2=9,
    stem_kernel_size3=11,
    stem_dropout=0.2917604668154306,
)

# Training recipe.
TRAIN = dict(
    learning_rate=0.0010405965473823274,  # peak learning rate of the one-cycle schedule
    weight_decay=1.287843673013275e-05,
    batch_size=64,
    epochs=300,
    patience=200,                          # early stopping on the validation loss
    grad_clip=1.0,
    huber_delta=0.5,
)

# Window decision (Section III of the paper): top-K mean of the per-sample P and S probabilities.
TOPK = 60

# MMI 5 threshold: PGA 42.3 cm/s^2, in model space ln(PGA[cm/s^2]) + 5 = 8.745.
PGA_MMI5_CMS2 = 42.3
