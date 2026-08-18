import torch.nn as nn

# argument order of nn.Conv1d / nn.Conv2d, used to retrieve arguments given
# positionally by the user
CONV_ARGS = (
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
)

# argument order of CausalConv2d
CAUSAL_CONV2D_ARGS = (
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "dilation",
    "bias",
    "padding_time",
    "padding_vert",
    "padding",
)


def get_padding(kernel_size, stride=1, dilation=1, mode="centered"):
    """
    Computes 'same' padding given a kernel size, stride an dilation.

    Parameters
    ----------

    kernel_size: int
        kernel_size of the convolution

    stride: int
        stride of the convolution

    dilation: int
        dilation of the convolution

    mode: str
        either "centered", "causal" or "anticausal"
    """
    if kernel_size == 1: return (0, 0)
    p = (kernel_size - 1) * dilation + 1
    if mode == "centered":
        p_right = p // 2
        p_left = (p - 1) // 2
    elif mode == "causal":
        p_right = 0
        p_left = p // 2 + (p - 1) // 2
    elif mode == "anticausal":
        p_right = p // 2 + (p - 1) // 2
        p_left = 0
    else:
        raise Exception(f"Padding mode {mode} is not valid")
    return (p_left, p_right)


def resolve_padding(padding, kernel_size, dilation=1):
    """
    Normalizes a padding specification into an explicit (left, right) pair.

    ``padding`` is either

    - a string: "same" (or "centered"), "causal", "anticausal", "valid" (or
      "none"), in which case the amount of padding is deduced from
      ``kernel_size`` and ``dilation`` using :func:`get_padding` ;
    - an int, applied on both sides ;
    - a (left, right) pair, used as is.
    """
    if padding is None:
        return (0, 0)

    if isinstance(padding, str):
        mode = padding.lower()
        if mode in ("valid", "none"):
            return (0, 0)
        if mode == "same":
            mode = "centered"
        return get_padding(kernel_size, dilation=dilation, mode=mode)

    if isinstance(padding, (list, tuple)):
        if len(padding) == 1:
            return (int(padding[0]), int(padding[0]))
        elif len(padding) == 2:
            return (int(padding[0]), int(padding[1]))
        raise ValueError(
            f"Padding {padding} must be an int or a (left, right) pair")

    return (int(padding), int(padding))


def axis_value(value, axis):
    """
    Retrieves the value of a per-axis convolution argument (kernel_size,
    stride, dilation), that can either be a scalar or a tuple.
    """
    if isinstance(value, (list, tuple)):
        return value[axis]
    return value


def peek_arg(args, kwargs, name, default=None, names=CONV_ARGS):
    """
    Reads an argument that may have been given positionally or as a keyword.
    """
    index = names.index(name)
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def take_arg(args, kwargs, name, default=None, names=CONV_ARGS):
    """
    Same as :func:`peek_arg`, but removes the argument from the call, replacing
    it with ``default`` when it was given positionally.
    """
    index = names.index(name)
    if len(args) > index:
        value = args[index]
        args[index] = default
        return value
    return kwargs.pop(name, default)


class Branches(nn.Module):

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = list(map(lambda x: x.cumulative_delay, self.branches))

        max_delay = max(delays)

        self.cumulative_delay = int(cumulative_delay * stride) + max_delay

    def forward(self, x):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        return outs
