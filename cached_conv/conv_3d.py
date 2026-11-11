"""
3D cached convolutions, for inputs shaped (batch, channels, depth, height, time).

Only the last (time) axis is streamed and cached. Depth and height are
always padded "same" (centered), exactly as height is in conv_2d.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_padding, normalization as _normalization
from .conv_1d import CachedPadding, CachedGroupNorm
from .conv_2d import AlignBranches2d

# CachedGroupNorm and the branch alignment are dimension agnostic.
CachedGroupNorm3d = CachedGroupNorm
AlignBranches3d = AlignBranches2d

MAX_BATCH_SIZE = 8


def _resolve_3d_padding(layer, padding):
    """F.pad list (time_l, time_r, h_top, h_bottom, d_front, d_back).

    ``padding`` only sets the time mode; depth and height are centered.
    """
    mode = "centered" if padding == "same" else padding
    if mode not in ("centered", "causal", "anticausal"):
        raise ValueError("padding must be 'same', 'centered', 'causal', or "
                         "'anticausal'")
    time = get_padding(layer.kernel_size[2], dilation=layer.dilation[2],
                       mode=mode)
    height = get_padding(layer.kernel_size[1], dilation=layer.dilation[1],
                         mode="centered")
    depth = get_padding(layer.kernel_size[0], dilation=layer.dilation[0],
                        mode="centered")
    return list(time + height + depth)


class CachedConv3d(nn.Conv3d):
    """Conv3d with cached temporal padding and centered depth/height padding.

    Padding is computed from kernel size and dilation. ``padding`` controls
    only time: "same" (or "centered"), "causal", or "anticausal". Input chunks
    must have fixed depth and height, and time lengths divisible by the
    temporal stride. ``normalization`` ("weight_norm" / True, or "identity" /
    None / False) optionally applies a weight normalization to the layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="same", *, dilation=1, groups=1, bias=True,
                 device=None, dtype=None, cumulative_delay=0,
                 max_batch_size=MAX_BATCH_SIZE, normalization=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias,
                         device=device, dtype=dtype)

        left, right, *spatial = _resolve_3d_padding(self, padding)
        self._pad = [0, 0] + spatial
        s = self.stride[2]
        cd = cumulative_delay
        stride_delay = (s - ((right + cd) % s)) % s
        self.cumulative_delay = (right + stride_delay + cd) // s

        self.cache = CachedPadding(left + right, max_batch_size=max_batch_size)
        self.downsampling_delay = CachedPadding(
            stride_delay, crop=True, max_batch_size=max_batch_size)

        _normalization(self, normalization)

    def _pad_input(self, x):
        x = self.downsampling_delay(x)
        x = self.cache(x)
        return F.pad(x, self._pad)

    def get_weight(self):
        """Kernel used by the convolution (overridden by variants)."""
        return self.weight

    def _conv(self, x, weight, bias: Optional[torch.Tensor], groups: int):
        return F.conv3d(x, weight, bias, self.stride, 0, self.dilation,
                         groups)

    def forward(self, x):
        return self._conv(self._pad_input(x), self.get_weight(), self.bias,
                          self.groups)


class Conv3d(nn.Conv3d):
    """Conv3d with automatic padding, always centered along depth/height.

    ``padding`` controls temporal padding: "same" (or "centered"), "causal",
    or "anticausal". Padding amounts follow kernel size and dilation.
    ``normalization`` optionally applies a weight normalization (see
    :func:`cached_conv.normalization`).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="same", *, dilation=1, groups=1, bias=True,
                 device=None, dtype=None, cumulative_delay=0,
                 max_batch_size=MAX_BATCH_SIZE, normalization=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias,
                         device=device, dtype=dtype)
        self._pad = _resolve_3d_padding(self, padding)
        self.cumulative_delay = 0
        _normalization(self, normalization)

    def _pad_input(self, x):
        return F.pad(x, self._pad)

    def get_weight(self):
        """Kernel used by the convolution (overridden by variants)."""
        return self.weight

    def _conv(self, x, weight, bias: Optional[torch.Tensor], groups: int):
        return F.conv3d(x, weight, bias, self.stride, 0, self.dilation,
                         groups)

    def forward(self, x):
        return self._conv(self._pad_input(x), self.get_weight(), self.bias,
                          self.groups)


class CachedConvTranspose3d(nn.ConvTranspose3d):
    """ConvTranspose3d with an overlap-add cache along time.

    ``padding`` is a regular (depth, height, time) padding. Depth and height
    padding are applied as usual; the time padding is handled by the cache,
    which requires ``kernel_size[2] == stride[2] + 2 * padding[2]`` for the
    output length to be exactly ``stride[2]`` times the input length.
    """

    def __init__(self, *args, **kwargs):
        cd = kwargs.pop("cumulative_delay", 0)
        max_batch_size = kwargs.pop("max_batch_size", MAX_BATCH_SIZE)
        super().__init__(*args, **kwargs)

        self.max_batch_size = max_batch_size
        self.initialized = 0
        self.time_pad = self.padding[2]
        self._padding = [self.padding[0], self.padding[1], 0]
        self.cumulative_delay = self.time_pad + cd * self.stride[2]

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        self.register_buffer(
            "cache",
            x.new_zeros(self.max_batch_size, *x.shape[1:-1],
                        2 * self.time_pad))
        self.initialized += 1

    def forward(self, x):
        x = F.conv_transpose3d(
            x,
            self.weight,
            None,
            self.stride,
            self._padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

        if not self.initialized:
            self.init_cache(x)

        padding = 2 * self.time_pad
        if padding > 0:
            x[..., :padding] += self.cache[:x.shape[0]]
            self.cache[:x.shape[0]].copy_(x[..., -padding:])
            x = x[..., :-padding]

        bias = self.bias
        if bias is not None:
            x = x + bias.view(-1, 1, 1, 1)

        return x


class ConvTranspose3d(nn.ConvTranspose3d):

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("cumulative_delay", 0)
        kwargs.pop("max_batch_size", None)
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0
