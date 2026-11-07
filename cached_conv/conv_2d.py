from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import is_cached
from .utils import get_padding

from .conv_1d import CachedPadding, CachedGroupNorm


try:
    from torch.fx import wrap as wrap
except ImportError:
    wrap = lambda fn: fn


CachedGroupNorm2d = CachedGroupNorm

MAX_BATCH_SIZE = 8

@wrap
def _resolve_2d_padding(layer, padding):
    mode = "centered" if padding == "same" else padding
    if mode not in ("centered", "causal", "anticausal"):
        raise ValueError("padding must be 'same', 'centered', 'causal', or 'anticausal'")
    time = get_padding(layer.kernel_size[1], dilation=layer.dilation[1],
                       mode=mode)
    vertical = get_padding(layer.kernel_size[0], dilation=layer.dilation[0],
                           mode="centered")
    return list(time + vertical)


class CachedConv2d(nn.Conv2d):
    """Conv2d with cached temporal padding and centered vertical padding.

    Padding is computed from kernel size and dilation. ``padding`` controls
    only time: "same" (or "centered"), "causal", or "anticausal". Input chunks must
    have fixed height and time lengths divisible by the temporal stride.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="same", *, dilation=1, groups=1, bias=True,
                 device=None, dtype=None, cumulative_delay=0,
                 max_batch_size=MAX_BATCH_SIZE):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias,
                         device=device, dtype=dtype)

        left, right, top, bottom = _resolve_2d_padding(self, padding)
        self._pad = [0, 0, top, bottom]
        s = self.stride[1]
        cd = cumulative_delay
        stride_delay = (s - ((right + cd) % s)) % s
        self.cumulative_delay = (right + stride_delay + cd) // s

        self.cache = CachedPadding(left + right, max_batch_size=max_batch_size)
        self.downsampling_delay = CachedPadding(
            stride_delay, crop=True, max_batch_size=max_batch_size)

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
    """Conv2d with automatic padding, always centered along height.

    ``padding`` controls temporal padding: "same" (or "centered"), "causal", or
    "anticausal". Padding amounts follow kernel size and dilation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="same", *, dilation=1, groups=1, bias=True,
                 device=None, dtype=None, cumulative_delay=0,
                 max_batch_size=MAX_BATCH_SIZE):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias,
                         device=device, dtype=dtype)
        self._pad = _resolve_2d_padding(self, padding)
        self.cumulative_delay = 0

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

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("cumulative_delay", 0)
        kwargs.pop("max_batch_size", None)
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0

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
