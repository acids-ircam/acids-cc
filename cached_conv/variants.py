"""
Convolution variants, available in 1D / 2D / 3D, cached or not:

- Windowed convolutions: the kernel is multiplied by a fixed window before
  the convolution (e.g. to obtain smooth, band-limited filters).
- Modulated convolutions: StyleGAN 2 weight (de)modulation, the kernel
  being scaled per sample and input channel by a style vector.

Both are mixins over the regular layers of conv_1d / conv_2d / conv_3d, so
padding, caching, cumulative delays and TorchScript work as usual.
"""
from typing import Optional

import torch
import torch.nn as nn

from .conv_1d import CachedConv1d, Conv1d
from .conv_2d import CachedConv2d, Conv2d
from .conv_3d import CachedConv3d, Conv3d

WINDOWS = ("hann", "hamming", "blackman", "bartlett", "kaiser")


def get_window(window, size):
    """1D window of ``size`` taps.

    ``window`` is either

    - a name among "hann", "hamming", "blackman", "bartlett", "kaiser". The
      window is computed over ``size + 2`` points and its two end points are
      dropped, so that no tap of the kernel is zeroed (a size 3 kernel gets
      ``[.5, 1, .5]`` with "hann", not ``[0, 1, 0]``),
    - None, "rect" or "boxcar": no windowing,
    - a callable ``f(size) -> Tensor``,
    - a 1D tensor / sequence of length ``size``.
    """
    if window is None or window in ("rect", "boxcar"):
        return torch.ones(size)
    if isinstance(window, str):
        if window not in WINDOWS:
            raise ValueError(f"unknown window {window}, must be one of "
                             f"{WINDOWS}, 'rect' or None")
        fn = getattr(torch, f"{window}_window")
        return fn(size + 2, periodic=False)[1:-1].float()
    if callable(window):
        w = window(size)
    else:
        w = torch.as_tensor(window, dtype=torch.float)
    if w.shape != (size, ):
        raise ValueError(f"window must have shape ({size}, ), got "
                         f"{tuple(w.shape)}")
    return w.float()


class WindowedMixin:
    """Multiplies the kernel by a separable window before convolving.

    ``window`` applies to every kernel axis (see :func:`get_window`). A
    *tuple* gives one window per axis, e.g. ``window=(None, "hann")`` to
    window only along time in 2D (lists / tensors are window values).
    """

    def __init__(self, *args, window="hann", **kwargs):
        super().__init__(*args, **kwargs)
        n = len(self.kernel_size)
        windows = window if isinstance(window, tuple) else (window, ) * n
        if len(windows) != n:
            raise ValueError(f"expected {n} windows (one per kernel axis), "
                             f"got {len(windows)}")
        full = torch.ones(())
        for axis, (w, size) in enumerate(zip(windows, self.kernel_size)):
            shape = [1] * n
            shape[axis] = size
            full = full * get_window(w, size).view(shape)
        # broadcast over (out_channels, in_channels // groups)
        self.register_buffer("window", full[None, None].to(self.weight))

    def get_weight(self):
        return self.weight * self.window


class ModulatedMixin:
    """StyleGAN 2 weight modulation / demodulation.

    ``forward(x, style)``: ``style`` is (batch, style_dim), mapped by a
    learned affine layer to one scale per input channel (bias initialized to
    1). Without ``style_dim``, ``style`` directly gives the scales, shaped
    (batch, in_channels). The kernel is scaled per sample and input channel,
    then (if ``demodulate``) normalized to unit norm per output channel.

    The style is applied to the whole current chunk: in streaming mode, it
    may change from one chunk to the next.
    """

    def __init__(self, *args, style_dim: Optional[int] = None,
                 demodulate: bool = True, eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.demodulate = demodulate
        self.eps = eps
        if style_dim is None:
            self.affine = nn.Identity()
        else:
            self.affine = nn.Linear(style_dim, self.in_channels)
            nn.init.ones_(self.affine.bias)

    def modulated_weight(self, style):
        """Modulated kernel, shaped
        ``(batch * out_channels, in_channels // groups, *kernel_size)``."""
        weight = self.get_weight()
        b = style.shape[0]
        g = self.groups
        out_c, in_g = weight.shape[0], weight.shape[1]
        kernel = list(weight.shape[2:])
        ones = [1] * len(kernel)

        scales = self.affine(style).reshape([b, g, 1, in_g] + ones)
        w = weight.reshape([1, g, out_c // g, in_g] + kernel) * scales
        if self.demodulate:
            dims = [3 + i for i in range(len(kernel) + 1)]
            w = w * torch.rsqrt(w.pow(2).sum(dims, keepdim=True) + self.eps)
        return w.reshape([b * out_c, in_g] + kernel)

    def forward(self, x, style):
        x = self._pad_input(x)
        b = x.shape[0]
        weight = self.modulated_weight(style)
        # samples are folded into groups: one convolution for the batch
        x = x.reshape([1, b * x.shape[1]] + list(x.shape[2:]))
        y = self._conv(x, weight, None, b * self.groups)
        y = y.reshape([b, -1] + list(y.shape[2:]))
        bias = self.bias
        if bias is not None:
            y = y + bias.reshape([1, -1] + [1] * (y.dim() - 2))
        return y


# ----- concrete classes ------------------------------------------------------


class WindowedConv1d(WindowedMixin, Conv1d):
    pass


class CachedWindowedConv1d(WindowedMixin, CachedConv1d):
    pass


class WindowedConv2d(WindowedMixin, Conv2d):
    pass


class CachedWindowedConv2d(WindowedMixin, CachedConv2d):
    pass


class WindowedConv3d(WindowedMixin, Conv3d):
    pass


class CachedWindowedConv3d(WindowedMixin, CachedConv3d):
    pass


class ModulatedConv1d(ModulatedMixin, Conv1d):
    pass


class CachedModulatedConv1d(ModulatedMixin, CachedConv1d):
    pass


class ModulatedConv2d(ModulatedMixin, Conv2d):
    pass


class CachedModulatedConv2d(ModulatedMixin, CachedConv2d):
    pass


class ModulatedConv3d(ModulatedMixin, Conv3d):
    pass


class CachedModulatedConv3d(ModulatedMixin, CachedConv3d):
    pass
