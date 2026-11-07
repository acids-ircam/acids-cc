"""
Upsampling compatible with cached (streaming) processing.

Inputs are shaped (batch, channels, *spatial, time). The time factor must be
an integer; spatial factors (2D / 3D) are applied as with nn.Upsample.

- "nearest": every time step is repeated. Causal, no cache, no delay.
- "linear" (bilinear / trilinear in 2D / 3D, align_corners=False): half of
  the interpolated samples need the next input step. The cached version
  keeps the last two input steps and delays the output by one input step
  (``scale`` output steps), giving exact linear interpolation across chunk
  boundaries.

Along time, linear interpolation is computed with exact per-phase weights
rather than F.interpolate, whose float32 source coordinates drift with the
signal length for factors that are not powers of 2 (~2e-4 after 1024 steps
for a factor 3). Eager and cached modules share this computation; the eager
one replicates the edges, as nn.Upsample does.
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_1d import CachedPadding

MAX_BATCH_SIZE = 8
_LINEAR_MODES = ("linear", "bilinear", "trilinear")


def _resolve_factors(scale_factor, n_dims):
    """-> (spatial factors, integer time factor)"""
    if isinstance(scale_factor, (tuple, list)):
        if len(scale_factor) != n_dims:
            raise ValueError(f"scale_factor must have {n_dims} values "
                             f"(*spatial, time), got {scale_factor}")
        factors = list(scale_factor)
    else:
        factors = [scale_factor] * n_dims
    time = factors[-1]
    if int(time) != time or time < 1:
        raise ValueError(f"time scale factor must be a positive integer, "
                         f"got {time}")
    return [float(f) for f in factors[:-1]], int(time)


def _resolve_mode(mode):
    if mode in _LINEAR_MODES:
        return "linear"
    if mode != "nearest":
        raise ValueError(f"mode must be 'nearest' or 'linear', got {mode}")
    return mode


def _linear_phases(s):
    """For output phase j of a factor ``s``, the source position is
    ``n + f_j`` with ``f_j = (j + .5) / s - .5`` (align_corners=False).
    In a signal extended by 2 leading samples and read one step late, it
    lies between ext[n + offset_j] and ext[n + offset_j + 1]."""
    offsets, fracs = [], []
    for j in range(s):
        f = (j + .5) / s - .5
        lo = math.floor(f)
        offsets.append(1 + lo)
        fracs.append(f - lo)
    return offsets, fracs


class _UpsampleBase(nn.Module):

    n_dims = 1

    def __init__(self, scale_factor, mode="nearest", cumulative_delay=0,
                 max_batch_size=MAX_BATCH_SIZE):
        super().__init__()
        spatial, time = _resolve_factors(scale_factor, self.n_dims)
        self.mode = _resolve_mode(mode)
        self.linear = self.mode == "linear"
        self.time_factor = time
        self.spatial_factor: List[float] = spatial + [1.]
        self.upsample_spatial = any(f != 1 for f in spatial)
        self.spatial_mode = "nearest" if not self.linear else \
            _LINEAR_MODES[self.n_dims - 1]
        offsets, fracs = _linear_phases(time)
        self.offsets: List[int] = offsets
        self.fracs: List[float] = fracs

    def _extend(self, x):
        """(..., T) -> (..., T + 2) signal the linear phases read from"""
        raise NotImplementedError

    def _upsample_time(self, x):
        s = self.time_factor
        if s == 1:
            return x
        if not self.linear:
            return x.repeat_interleave(s, dim=-1)
        length = x.shape[-1]
        ext = self._extend(x)
        phases: List[torch.Tensor] = []
        for j in range(s):
            o = self.offsets[j]
            a = self.fracs[j]
            lo = ext[..., o:o + length]
            hi = ext[..., o + 1:o + 1 + length]
            phases.append(lo + a * (hi - lo))
        y = torch.stack(phases, -1)
        return y.reshape(list(y.shape[:-2]) + [length * s])

    def forward(self, x):
        x = self._upsample_time(x)
        if self.upsample_spatial:
            if self.linear:
                x = F.interpolate(x, scale_factor=self.spatial_factor,
                                  mode=self.spatial_mode, align_corners=False)
            else:
                x = F.interpolate(x, scale_factor=self.spatial_factor,
                                  mode="nearest")
        return x


class _Upsample(_UpsampleBase):
    """Eager upsampling, equivalent to nn.Upsample(align_corners=False)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0

    def _extend(self, x):
        # replicated edges; read with offsets as if one step late
        return torch.cat([x[..., :1], x, x[..., -1:]], -1)


class _CachedUpsample(_UpsampleBase):

    def __init__(self, scale_factor, mode="nearest", cumulative_delay=0,
                 max_batch_size=MAX_BATCH_SIZE):
        super().__init__(scale_factor, mode, cumulative_delay, max_batch_size)
        delayed = self.linear and self.time_factor > 1
        self.time_delay = self.time_factor if delayed else 0
        self.cumulative_delay = int(cumulative_delay * self.time_factor) \
            + self.time_delay
        # an unused CachedPadding would never allocate its buffer, which
        # TorchScript then cannot compile
        self.cache = CachedPadding(2, max_batch_size=max_batch_size) \
            if delayed else nn.Identity()

    def _extend(self, x):
        # last two steps of the previous chunk, output one step late
        return self.cache(x)


class Upsample1d(_Upsample):
    n_dims = 1


class Upsample2d(_Upsample):
    n_dims = 2


class Upsample3d(_Upsample):
    n_dims = 3


class CachedUpsample1d(_CachedUpsample):
    n_dims = 1


class CachedUpsample2d(_CachedUpsample):
    n_dims = 2


class CachedUpsample3d(_CachedUpsample):
    n_dims = 3
