"""PWaveMamba: convolutional stem + stacked selective SSM (Mamba) layers + three task heads.

Module and parameter names are those of the training code, so the released checkpoints load with
`strict=True`.

Input  x: (B, 3, L) three-component acceleration window after `segment_preprocess_data`.
Output (magnitude (B,), intensity (B,), detection logits (B, L, 3) for {noise, P, S}).
The intensity target is ln(RotD50 PGA [cm/s^2]) + 5.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mamba_layer(d_model, d_state, d_conv, expand, backend="auto"):
    """One Mamba block. backend: 'mamba_ssm' (CUDA kernels), 'reference' (pure PyTorch) or 'auto'
    ('mamba_ssm' if it is installed and a GPU is available, otherwise 'reference')."""
    if backend == "auto":
        backend = "reference"
        if torch.cuda.is_available():
            try:
                import mamba_ssm  # noqa: F401
                backend = "mamba_ssm"
            except ImportError:
                pass
    if backend == "mamba_ssm":
        from mamba_ssm import Mamba
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if backend == "reference":
        from .mamba_ref import MambaRef
        return MambaRef(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    raise ValueError(f"unknown Mamba backend {backend!r}")


class MultiScaleConvStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, conv1_filters=32, conv2_filters=32, conv3_filters=32,
                 kernel_size1=4, kernel_size2=4, kernel_size3=4, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, conv1_filters, kernel_size1, padding="same")
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size2, padding="same")
        self.conv3 = nn.Conv1d(conv2_filters, conv3_filters, kernel_size3, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Conv1d(conv3_filters, out_channels, kernel_size=1)  # 1x1 projection to d_model
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):                      # (B, 3, L)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.proj(x)
        return self.norm(x.permute(0, 2, 1))   # (B, L, d_model)


class IntensityScalarHead(nn.Module):
    def __init__(self, input_dim, fc1_neurons, fc2_neurons, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, fc1_neurons), nn.SiLU(), nn.LayerNorm(fc1_neurons),
                                    nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(fc1_neurons, fc2_neurons), nn.SiLU(), nn.LayerNorm(fc2_neurons),
                                    nn.Dropout(dropout))
        self.skip_proj = nn.Linear(fc1_neurons, fc2_neurons) if fc1_neurons != fc2_neurons else nn.Identity()
        self.layer3 = nn.Linear(fc2_neurons, 1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1) + self.skip_proj(x1)
        return self.layer3(x2)


class PWaveMamba(nn.Module):
    def __init__(self, window_size, d_model=64, n_layers=2, d_state=16, d_conv=4, expand=2, fc1_neurons=32,
                 fc2_neurons=16, dropout=0.2, mag_dropout=None, int_dropout=None, det_dropout=None,
                 stem_conv1_filters=32, stem_conv2_filters=32, stem_conv3_filters=32, stem_kernel_size=4,
                 stem_kernel_size1=None, stem_kernel_size2=None, stem_kernel_size3=None, stem_dropout=None,
                 backend="auto"):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        k1 = stem_kernel_size if stem_kernel_size1 is None else stem_kernel_size1
        k2 = stem_kernel_size if stem_kernel_size2 is None else stem_kernel_size2
        k3 = stem_kernel_size if stem_kernel_size3 is None else stem_kernel_size3
        stem_drop = dropout if stem_dropout is None else stem_dropout
        mag_drop = dropout if mag_dropout is None else mag_dropout
        int_drop = dropout if int_dropout is None else int_dropout
        det_drop = dropout if det_dropout is None else det_dropout

        self.input_proj = MultiScaleConvStem(3, d_model, stem_conv1_filters, stem_conv2_filters, stem_conv3_filters,
                                             k1, k2, k3, dropout=stem_drop)
        self.mamba_layers = nn.ModuleList([mamba_layer(d_model, d_state, d_conv, expand, backend)
                                           for _ in range(n_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.magnitude_scalar_head = nn.Sequential(
            nn.Linear(d_model * 2, fc1_neurons), nn.SiLU(), nn.LayerNorm(fc1_neurons), nn.Dropout(mag_drop),
            nn.Linear(fc1_neurons, fc2_neurons), nn.SiLU(), nn.LayerNorm(fc2_neurons), nn.Dropout(mag_drop),
            nn.Linear(fc2_neurons, 1))
        self.intensity_scalar_head = IntensityScalarHead(d_model * 2, fc1_neurons, fc2_neurons, dropout=int_drop)
        self.detection_head = nn.Sequential(  # per sample, 3 classes: noise, P, S
            nn.Linear(d_model * 2, 32), nn.ReLU(), nn.Dropout(det_drop), nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 3))

    def backbone(self, x):
        x = self.input_proj(x)
        for i, layer in enumerate(self.mamba_layers):  # pre-norm residual blocks
            x = x + self.dropout(layer(self.layer_norms[i](x)))
        return x

    def forward(self, x):
        x = self.backbone(x)
        x_max, _ = x.max(dim=1)
        x_mean = x.mean(dim=1)
        ts_input = torch.cat([x, x_max.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)  # (B, L, 2*d_model)
        detection_ts = self.detection_head(ts_input).clamp(-50, 50)

        # P-conditioned pooling for intensity: average weighted by the per-sample P probability
        event_mask = F.softmax(detection_ts, dim=-1)[:, :, 1].unsqueeze(-1)              # (B, L, 1)
        x_weighted_avg = (x * event_mask).sum(dim=1) / torch.clamp(event_mask.sum(dim=1), min=1e-3)

        magnitude = self.magnitude_scalar_head(torch.cat([x_max, x_mean], dim=-1)).squeeze(-1)
        intensity = self.intensity_scalar_head(torch.cat([x_max, x_weighted_avg], dim=-1)).squeeze(-1)
        return magnitude, intensity, detection_ts


PWaveMambaNew = PWaveMamba  # name used in the original training code
