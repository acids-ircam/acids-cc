"""
Sample rate conversion compatible with cached (streaming) processing.

Converts a signal shaped ``(batch, channels, time)`` from ``orig_sr`` to
``target_sr``. Any pair of integer rates is supported: with ``L / M`` the
reduced ratio ``target_sr / orig_sr``, the conversion is a polyphase filter
with ``L`` phases, applied with a stride of ``M`` input samples (as in
:func:`torchaudio.functional.resample`).

Each phase is a Kaiser-windowed sinc, low-pass at ``rolloff`` times the lowest
of the two Nyquist frequencies, normalized to a unit DC gain. The filter is
designed in pure torch: no scipy / torchaudio dependency.

- ``Resample1d`` (eager): zero-delay ("centered") filtering of a whole signal.
  Any input length is accepted; the output has ``ceil(T * L / M)`` samples.
- ``CachedResample1d`` (streaming): the same filter made causal. Inputs must
  have a length multiple of ``M``, so that every chunk gives exactly
  ``T * L / M`` output samples. The output is delayed by ``time_delay``
  output samples w.r.t. the eager one.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_1d import CachedPadding

MAX_BATCH_SIZE = 64


def kaiser_beta(attenuation: float) -> float:
    """Kaiser window beta giving a stopband ``attenuation`` (dB, positive)."""
    a = float(attenuation)
    if a > 50:
        return 0.1102 * (a - 8.7)
    if a > 21:
        return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    return 0.


def resample_ratio(orig_sr: int, target_sr: int):
    """-> (L, M): reduced up / down factors, ``target_sr / orig_sr = L / M``"""
    if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
        raise ValueError(f"sampling rates must be integers, "
                         f"got {orig_sr} and {target_sr}")
    orig_sr, target_sr = int(orig_sr), int(target_sr)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"sampling rates must be positive, "
                         f"got {orig_sr} and {target_sr}")
    g = math.gcd(orig_sr, target_sr)
    return target_sr // g, orig_sr // g


def resample_kernel(up: int,
                    down: int,
                    lowpass_filter_width: int = 32,
                    rolloff: float = .9,
                    beta: float = kaiser_beta(100)):
    """
    Polyphase kernel of a ``up / down`` resampler.

    Returns ``(kernel, left)``: ``kernel`` is shaped ``(up, 1, K)``, phase
    ``j`` computing the output sample at input time ``n * down + j * down /
    up`` from input samples ``n * down - left + k``, k in [0, K). ``K - down``
    is a multiple of ``down``, so that the right context ``K - down - left``
    spans a whole number of strides.
    """
    # cutoff, in cycles per input sample
    cutoff = .5 * rolloff * min(1., up / down)
    # half width of the filter, in input samples
    width = int(math.ceil(lowpass_filter_width / (2 * cutoff)))
    left = width
    right = int(math.ceil((width + 1) / down)) * down
    size = left + down + right

    k = torch.arange(size, dtype=torch.float64)
    phase = torch.arange(up, dtype=torch.float64) * down / up
    t = k[None] - left - phase[:, None]  # (up, K), in input samples

    h = 2 * cutoff * torch.sinc(2 * cutoff * t)
    r = (t / width).clamp(-1, 1)
    window = torch.special.i0(beta * torch.sqrt(1 - r**2)) / \
        torch.special.i0(torch.tensor(beta, dtype=torch.float64))
    h = h * window * (t.abs() <= width)
    h = h / h.sum(-1, keepdim=True)
    return h.float().unsqueeze(1), left


class _ResampleBase(nn.Module):

    def __init__(self,
                 orig_sr: int,
                 target_sr: int,
                 lowpass_filter_width: int = 32,
                 rolloff: float = .9,
                 beta: Optional[float] = None,
                 attenuation: float = 100.,
                 cumulative_delay: int = 0,
                 max_batch_size: int = MAX_BATCH_SIZE):
        super().__init__()
        up, down = resample_ratio(orig_sr, target_sr)
        if beta is None:
            beta = kaiser_beta(attenuation)
        kernel, left = resample_kernel(up, down, lowpass_filter_width,
                                       rolloff, beta)
        self.orig_sr = int(orig_sr)
        self.target_sr = int(target_sr)
        self.up = up
        self.down = down
        self.left = left
        self.right = kernel.shape[-1] - down - left
        self.identity = up == down
        self.register_buffer("kernel", kernel)

    def _resample(self, x):
        """x is padded so that its length is ``n * down + K - down``"""
        shape = x.shape
        x = x.reshape(-1, 1, shape[-1])
        y = F.conv1d(x, self.kernel.to(x.dtype), stride=self.down)
        y = y.transpose(1, 2).reshape(list(shape[:-1]) + [-1])
        return y


class Resample1d(_ResampleBase):
    """Eager resampling: zero-delay, whole signal at once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0
        self.time_delay = 0

    def forward(self, x):
        if self.identity:
            return x
        length = x.shape[-1]
        extra = (self.down - length % self.down) % self.down
        x = F.pad(x, [self.left, self.right + extra])
        y = self._resample(x)
        return y[..., :(length * self.up + self.down - 1) // self.down]


class CachedResample1d(_ResampleBase):
    """Streaming resampling: causal, chunk length multiple of ``down``."""

    def __init__(self,
                 orig_sr: int,
                 target_sr: int,
                 lowpass_filter_width: int = 32,
                 rolloff: float = .9,
                 beta: Optional[float] = None,
                 attenuation: float = 100.,
                 cumulative_delay: int = 0,
                 max_batch_size: int = MAX_BATCH_SIZE):
        super().__init__(orig_sr, target_sr, lowpass_filter_width, rolloff,
                         beta, attenuation, cumulative_delay, max_batch_size)
        if self.identity:
            self.time_delay = 0
            self.cache = nn.Identity()
        else:
            self.time_delay = self.right // self.down * self.up
            self.cache = CachedPadding(self.left + self.right,
                                       max_batch_size=max_batch_size)
        self.cumulative_delay = \
            int(cumulative_delay * self.up // self.down) + self.time_delay

    def forward(self, x):
        if self.identity:
            return x
        if x.shape[-1] % self.down:
            raise ValueError("CachedResample1d: chunk length (" +
                             str(x.shape[-1]) + ") must be a multiple of " +
                             str(self.down))
        return self._resample(self.cache(x))
