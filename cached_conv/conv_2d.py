from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import is_cached
from .utils import resolve_padding, normalization as _normalization

from .conv_1d import CachedPadding, CachedGroupNorm


try:
    from torch.fx import wrap as wrap
except ImportError:
    wrap = lambda fn: fn


CachedGroupNorm2d = CachedGroupNorm

MAX_BATCH_SIZE = 8

_PADDING_MODES = ("same", "centered", "causal", "anticausal", "valid", "none")


def _resolve_axis_padding(padding, kernel_size, dilation, name):
    if isinstance(padding, str) and padding.lower() not in _PADDING_MODES:
        raise ValueError(f"{name} must be an int, a (before, after) pair, or "
                         f"one of {_PADDING_MODES}, got {padding!r}")
    return resolve_padding(padding, kernel_size, dilation)


def _resolve_2d_padding(layer, padding, padding_time, padding_vert):
    """Returns the (left, right, top, bottom) padding of a 2d layer.

    ``padding`` and ``padding_time`` are synonyms (time = last axis) and cannot
    be given together. Each of ``padding_time`` / ``padding_vert`` is either a
    mode ("same" / "centered", "causal", "anticausal", "valid" / "none"), an
    int (both sides) or a (before, after) pair.
    """
    if padding is not None and padding_time is not None:
        raise ValueError("padding and padding_time are equivalent, "
                         "only one of them can be given")
    if padding_time is None:
        padding_time = "same" if padding is None else padding
    if padding_vert is None:
        padding_vert = "same"
    time = _resolve_axis_padding(padding_time, layer.kernel_size[1],
                                 layer.dilation[1], "padding_time")
    vertical = _resolve_axis_padding(padding_vert, layer.kernel_size[0],
                                     layer.dilation[0], "padding_vert")
    return list(time + vertical)


class CachedConv2d(nn.Conv2d):
    """Conv2d with cached temporal padding and explicit vertical padding.

    ``padding`` (or its synonym ``padding_time``) sets the temporal (last axis)
    padding, ``padding_vert`` the vertical (height) one. Both accept a mode
    ("same" / "centered", "causal", "anticausal", "valid" / "none") whose
    amount follows kernel size and dilation, an int (both sides) or a
    (before, after) pair. Time defaults to "same", height to "same".

    ``normalization`` ("weight_norm" / True, or "identity" / None / False)
    optionally applies a weight normalization to the layer.

    Input chunks must have fixed height and time lengths divisible by the
    temporal stride.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, *, padding_time=None, padding_vert=None,
                 dilation=1, groups=1, bias=True, device=None, dtype=None,
                 cumulative_delay=0, max_batch_size=MAX_BATCH_SIZE,
                 normalization=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias,
                         device=device, dtype=dtype)

        left, right, top, bottom = _resolve_2d_padding(
            self, padding, padding_time, padding_vert)
        self._pad = [0, 0, top, bottom]
        s = self.stride[1]
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
        return F.conv2d(x, weight, bias, self.stride, 0, self.dilation,
                         groups)

    def forward(self, x):
        return self._conv(self._pad_input(x), self.get_weight(), self.bias,
                          self.groups)


class CachedConvTranspose2d(nn.ConvTranspose2d):
    """
    Implementation of a ConvTranspose2d operation with cached padding
    """

    def __init__(self, *args, **kwargs):
        cd = kwargs.pop("cumulative_delay", 0)
        max_batch_size = kwargs.pop("max_batch_size", MAX_BATCH_SIZE)
        # the cache always crops on the right (see ConvTranspose2d)
        kwargs.pop("crop_right", None)
        super().__init__(*args, **kwargs)

        self.max_batch_size = max_batch_size
        self.initialized = 0

        if is_cached():
            stride = self.stride[1]
            self.cumulative_delay = self.padding[1] + cd * stride
            self._padding = [self.padding[0], 0]
            self.time_pad = self.padding[1]
            self.use_cache = True
        else:
            self.cumulative_delay = 0
            self._padding = [self.padding[0], 0]
            self.time_pad = self.padding[1]
            self.use_cache = False

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, h, _ = x.shape
        self.register_buffer(
            "cache",
            torch.zeros(
                self.max_batch_size,
                c,
                h,
                2 * self.time_pad,
            ).to(x))
        self.initialized += 1

    def forward(self, x):
        x = F.conv_transpose2d(
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

        if self.use_cache:
            x[..., :padding] += self.cache[:x.shape[0]]
            self.cache[:x.shape[0]].copy_(x[..., -padding:])

        if self.time_pad > 0:
            x = x[..., :-padding]

        bias = self.bias
        if bias is not None:
            x = x + bias.unsqueeze(-1).unsqueeze(-1)

        return x


class Conv2d(nn.Conv2d):
    """Conv2d with automatic padding.

    Same padding arguments as :class:`CachedConv2d`: ``padding`` (or its
    synonym ``padding_time``) for time, ``padding_vert`` for height, and an
    optional ``normalization``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, *, padding_time=None, padding_vert=None,
                 dilation=1, groups=1, bias=True, device=None, dtype=None,
                 cumulative_delay=0, max_batch_size=MAX_BATCH_SIZE,
                 normalization=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias,
                         device=device, dtype=dtype)
        self._pad = _resolve_2d_padding(self, padding, padding_time,
                                        padding_vert)
        self.cumulative_delay = 0
        _normalization(self, normalization)

    def _pad_input(self, x):
        return F.pad(x, self._pad)

    def get_weight(self):
        """Kernel used by the convolution (overridden by variants)."""
        return self.weight

    def _conv(self, x, weight, bias: Optional[torch.Tensor], groups: int):
        return F.conv2d(x, weight, bias, self.stride, 0, self.dilation,
                         groups)

    def forward(self, x):
        return self._conv(self._pad_input(x), self.get_weight(), self.bias,
                          self.groups)


class ConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d, non-cached counterpart of :class:`CachedConvTranspose2d`.

    ``crop_right`` (default ``True``) sets how the temporal padding is removed
    from the output:

    - ``True``: ``2 * padding[1]`` samples are cropped at the end only, as
      the cached layer does. The output is then identical to the streamed one
      (and to RAVE's legacy implementation), i.e. delayed by ``padding[1]``
      w.r.t. torch's centered cropping; ``cumulative_delay`` reports it.
    - ``False``: torch's behavior, ``padding[1]`` cropped on each side.

    Height padding is always cropped on both sides.
    """

    def __init__(self, *args, crop_right: bool = True, **kwargs) -> None:
        cd = kwargs.pop("cumulative_delay", 0)
        kwargs.pop("max_batch_size", None)
        super().__init__(*args, **kwargs)
        self.crop_right = crop_right
        if crop_right:
            self._padding = [self.padding[0], 0]
            self._crop = 2 * self.padding[1]
            self.cumulative_delay = self.padding[1] + cd * self.stride[1]
        else:
            self._padding = [self.padding[0], self.padding[1]]
            self._crop = 0
            self.cumulative_delay = 0

    def forward(self, x):
        x = F.conv_transpose2d(x, self.weight, self.bias, self.stride,
                               self._padding, self.output_padding,
                               self.groups, self.dilation)
        if self._crop > 0:
            x = x[..., :-self._crop]
        return x

class AlignBranches2d(nn.Module):

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = list(map(lambda x: x.cumulative_delay, self.branches))

        max_delay = max(delays)

        self.paddings = nn.ModuleList([
            CachedPadding(p, crop=True)
            for p in map(lambda f: max_delay - f, delays)
        ])

        self.cumulative_delay = int(cumulative_delay * stride) + max_delay

    def forward(self, x):
        outs = []
        for branch, pad in zip(self.branches, self.paddings):
            delayed_x = pad(x)
            outs.append(branch(delayed_x))
        return outs
