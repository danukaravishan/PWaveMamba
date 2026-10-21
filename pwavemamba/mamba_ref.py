"""Pure-PyTorch reference implementation of one Mamba (selective SSM) block.

Used when the CUDA package `mamba_ssm` is not available or the model runs on a CPU. Parameter names
and shapes are identical to `mamba_ssm.Mamba` (v1.2), so the released checkpoints load unchanged.
The scan is evaluated step by step; this is the "pure-PyTorch export" whose CPU latency is reported
in the paper (Table IV). It is meant for inference: training from scratch should use `mamba_ssm`,
whose parameter initialization (e.g. of dt_proj) this module does not replicate.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaRef(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        """x: (B, L, d_model) -> (B, L, d_model)."""
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)  # causal depthwise convolution
        u = F.silu(u)
        dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                                # (d_inner, N)
        dA = torch.exp(dt.unsqueeze(-1) * A)                              # (B, L, d_inner, N)
        dBu = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)         # (B, L, d_inner, N)
        h = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device, dtype=dA.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.D
        return self.out_proj(y * F.silu(z))
