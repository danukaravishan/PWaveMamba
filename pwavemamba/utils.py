import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def env_info():
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import mamba_ssm
        info["mamba_ssm"] = getattr(mamba_ssm, "__version__", "?")
    except Exception:
        info["mamba_ssm"] = None
    return info


def load_checkpoint(model, path, device="cpu", strict=True):
    """Load a released or newly trained checkpoint ({'model_state_dict': ...} or a raw state dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd, strict=strict)
    return ck


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
