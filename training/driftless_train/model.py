"""The speed & heading-change regressor -- a small dilated temporal CNN.

Design constraints that shaped this, in order:
  1. It runs on a phone at 10 Hz inside a live navigation loop, so it must be
     small and cheap: ~4 dilated conv blocks, tens of thousands of parameters.
  2. It must be causal. A window ending now may not look forward.
  3. Its outputs go straight into role 02's EKF, so they are in SI units
     (m/s and rad). Input normalisation and output de-normalisation are baked
     into the graph as constant buffers, so the phone and the C++ edge engine
     cannot disagree with training about scaling -- there is nothing for them to
     reimplement.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalBlock(nn.Module):
    """Dilated causal conv -> norm -> GELU, with a residual path."""

    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.norm = nn.GroupNorm(4, ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad only: output at t depends on inputs <= t.
        y = nn.functional.pad(x, (self.pad, 0))
        y = self.drop(self.act(self.norm(self.conv(y))))
        return x + y


class SpeedHeadingTCN(nn.Module):
    """(B, C, W) IMU context -> (B, 3) = [mean speed m/s, dpsi rad, dv m/s].

    Dilations reach 16 so the receptive field covers an 8 s context at 10 Hz:
    1 + 2*(1+2+4+8+16) = 63 samples of history feed the final timestep.
    """

    def __init__(self, in_channels: int, width: int = 48, kernel: int = 3,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
                 dropout: float = 0.1, n_out: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.n_out = n_out
        self.stem = nn.Conv1d(in_channels, width, 1)
        self.blocks = nn.Sequential(
            *[CausalBlock(width, kernel, d, dropout) for d in dilations])
        self.head = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, n_out, 1))

        # Normalisation constants, overwritten by set_stats() before training.
        self.register_buffer("x_mean", torch.zeros(in_channels))
        self.register_buffer("x_std", torch.ones(in_channels))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))

    @torch.no_grad()
    def set_stats(self, stats: dict) -> None:
        self.x_mean.copy_(torch.tensor(stats["x_mean"], dtype=torch.float32))
        self.x_std.copy_(torch.tensor(stats["x_std"], dtype=torch.float32))
        self.y_mean.copy_(torch.tensor(stats["y_mean"], dtype=torch.float32))
        self.y_std.copy_(torch.tensor(stats["y_std"], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.x_mean[None, :, None]) / self.x_std[None, :, None]
        h = self.blocks(self.stem(x))
        y = self.head(h)[:, :, -1]            # causal: read the last timestep only
        return y * self.y_std[None, :] + self.y_mean[None, :]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SpeedHeadingBaseline(nn.Module):
    """Non-learned reference: the honest thing to beat.

    Speed comes from integrating horizontal acceleration over the window (which
    drifts, as expected), heading change from integrating gyro about vertical
    (which is quite good on its own). If the TCN cannot beat this, the learned
    component is not earning its place in the report.
    """

    def __init__(self, acc_horiz_idx: int, gyro_vert_idx: int, dt: float = 0.1):
        super().__init__()
        self.acc_i, self.gyro_i, self.dt = acc_horiz_idx, gyro_vert_idx, dt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        speed = x[:, self.acc_i, :].abs().cumsum(dim=-1)[:, -1] * self.dt
        dpsi = x[:, self.gyro_i, :].sum(dim=-1) * self.dt
        return torch.stack([speed, dpsi], dim=-1)
