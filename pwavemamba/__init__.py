"""PWaveMamba: multi-task selective state-space model for single-station earthquake early warning."""
from .config import MODEL_HPARAMS, TRAIN, WINDOWS, DEFAULT_WINDOW, TOPK
from .model import PWaveMamba, PWaveMambaNew
from .variants import VARIANTS, build_model, loss_weights
from .preprocessing import segment_preprocess_data
from .decision import predict, topk_conf, classify
from .utils import load_checkpoint, set_seed

__version__ = "1.1.0"
