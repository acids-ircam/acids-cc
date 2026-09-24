import torch
import torch.nn as nn

from .utils import (get_padding, resolve_padding, axis_value, peek_arg,
                    take_arg)

MAX_BATCH_SIZE = 64


class CachedSequential(nn.Sequential):
    """
    Sequential operations with future compensation tracking
    """

    def __init__(self, *args, **kwargs):
        cumulative_delay = kwargs.pop("cumulative_delay", 0)
        stride = kwargs.pop("stride", 1)
        super().__init__(*args, **kwargs)

        self.cumulative_delay = int(cumulative_delay) * stride

        last_delay = 0
        for i in range(1, len(self) + 1):
            try:
                last_delay = self[-i].cumulative_delay
                break
            except AttributeError:
                pass
        self.cumulative_delay += last_delay


class Sequential(CachedSequential):
    pass


@torch.fx.wrap
class CachedPadding(nn.Module):
    """
    Cached Padding implementation, replace zero padding with the end of
    the previous tensor.
    """

    def __init__(self, padding, crop=False, max_batch_size=MAX_BATCH_SIZE):
        super().__init__()
        self.initialized = 0
        self.padding = padding
        self.crop = crop
        self.max_batch_size = max_batch_size

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        shape = (self.max_batch_size, *x.shape[1:-1], self.padding)
        self.register_buffer("pad", x.new_zeros(shape))
        self.initialized += 1

    def forward(self, x):
        if not self.initialized:
            self.init_cache(x)

        if self.padding:
            x = torch.cat([self.pad[:x.shape[0]], x], -1)
            self.pad[:x.shape[0]].copy_(x[..., -self.padding:])

            if self.crop:
                x = x[..., :-self.padding]

        return x


class CachedConv1d(nn.Conv1d):
    """
    Implementation of a Conv1d operation with cached padding
    """

    def __init__(self, *args, **kwargs):
        args = list(args)
        cumulative_delay = kwargs.pop("cumulative_delay", 0)
        max_batch_size = kwargs.pop("max_batch_size", MAX_BATCH_SIZE)

        padding = take_arg(args, kwargs, "padding", 0)
        kernel_size = axis_value(peek_arg(args, kwargs, "kernel_size"), 0)
        dilation = axis_value(peek_arg(args, kwargs, "dilation", 1), 0)
        padding = resolve_padding(padding, kernel_size, dilation)

        super().__init__(*args, **kwargs)

        r_pad = padding[1]
        padding = padding[0] + padding[1]

        s = self.stride[0]
        cd = cumulative_delay

        stride_delay = (s - ((r_pad + cd) % s)) % s

        self.cumulative_delay = (r_pad + stride_delay + cd) // s

        self.cache = CachedPadding(padding, max_batch_size=max_batch_size)
        self.downsampling_delay = CachedPadding(stride_delay, crop=True, max_batch_size=max_batch_size)

    def forward(self, x):
        x = self.downsampling_delay(x)
        x = self.cache(x)
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CachedConvTranspose1d(nn.ConvTranspose1d):
    """
    Implementation of a ConvTranspose1d operation with cached padding
    """

    def __init__(self, *args, **kwargs):
        cd = kwargs.pop("cumulative_delay", 0)
        max_batch_size = kwargs.pop("max_batch_size", MAX_BATCH_SIZE)
        super().__init__(*args, **kwargs)
        stride = self.stride[0]
        self.initialized = 0
        self.max_batch_size = max_batch_size
        self.cumulative_delay = self.padding[0] + cd * stride

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, _ = x.shape
        self.register_buffer(
            "cache",
            torch.zeros(
                self.max_batch_size,
                c,
                2 * self.padding[0],
            ).to(x))
        self.initialized += 1

    def forward(self, x):
        x = nn.functional.conv_transpose1d(
            x,
            self.weight,
            None,
            self.stride,
            0,
            self.output_padding,
            self.groups,
            self.dilation,
        )

        if not self.initialized:
            self.init_cache(x)

        padding = 2 * self.padding[0]

        x[..., :padding] += self.cache[:x.shape[0]]
        self.cache[:x.shape[0]].copy_(x[..., -padding:])

        x = x[..., :-padding]

        bias = self.bias
        if bias is not None:
            x = x + bias.unsqueeze(-1)
        return x


class ConvTranspose1d(nn.ConvTranspose1d):

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("cumulative_delay", 0)
        kwargs.pop("max_batch_size", None)
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0


class Conv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        args = list(args)
        kwargs.pop("cumulative_delay", 0)
        kwargs.pop("max_batch_size", None)

        padding = take_arg(args, kwargs, "padding", 0)
        kernel_size = axis_value(peek_arg(args, kwargs, "kernel_size"), 0)
        dilation = axis_value(peek_arg(args, kwargs, "dilation", 1), 0)
        self._pad = list(resolve_padding(padding, kernel_size, dilation))

        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0

    def forward(self, x):
        x = nn.functional.pad(x, self._pad)
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CachedGroupNorm(nn.GroupNorm):
    """Block-streaming GroupNorm for [batch, channels, ..., time] inputs.

    Statistics use retained history plus the entire current chunk. Only
    the current chunk is returned, so results depend on chunk boundaries
    and are not equivalent to GroupNorm over a full recording. This is
    causal at chunk granularity, not at individual sample granularity.

    ``history_size`` counts previous samples; "automatic" uses the first
    chunk length. Startup history is zero-filled. Zero disables history.
    Batch slots must retain their stream identity; call ``reset_cache``
    before processing a new stream. Spatial dimensions must stay fixed.
    History is detached between calls.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                 device=None, dtype=None, *, history_size="automatic",
                 max_batch_size=MAX_BATCH_SIZE):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine,
                         device=device, dtype=dtype)
        
        automatic = history_size == "automatic"

        self.automatic_history = automatic
        self.history_size = 0 if automatic else history_size
        self.max_batch_size = max_batch_size
        self.initialized = False
        self.register_buffer("history", torch.empty(0, device=device,
                                                    dtype=dtype),
                             persistent=False)

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        if self.automatic_history:
            self.history_size = x.shape[-1]
        shape = (self.max_batch_size, *x.shape[1:-1], self.history_size)
        self.history = x.new_zeros(shape)
        self.initialized = True

    @torch.jit.export
    def reset_cache(self):
        self.history.zero_()

    def forward(self, x):
        if not self.initialized:
            self.init_cache(x)
            
        size = x.shape[-1]

        if self.history_size:
            x = torch.cat([self.history[:x.shape[0]], x], dim=-1)
            self.history[:x.shape[0]].copy_(
                x[..., -self.history_size:].detach())
            
        x = nn.functional.group_norm(x, self.num_groups, self.weight,
                                     self.bias, self.eps)
        return x[..., -size:]


# Compatibility name for the shared implementation.
CachedGroupNorm1d = CachedGroupNorm


class AlignBranches1d(nn.Module):

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
