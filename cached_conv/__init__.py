import torch

USE_CACHED_CONV = False


def use_cached_conv(state: bool):
    global USE_CACHED_CONV
    USE_CACHED_CONV = state


def is_cached():
    global USE_CACHED_CONV
    return USE_CACHED_CONV


from .utils import Branches
from .conv_1d import (get_padding, CachedSequential, Sequential,
                      CachedPadding1d, CachedConv1d, CachedConvTranspose1d)
from .conv_1d import Conv1d as _Conv1d
from .conv_1d import ConvTranspose1d as _ConvTranspose1d
from .conv_1d import AlignBranches1d as _AlignBranches1d
from .conv_2d import (CausalConv2d, CachedPadding2d,
                      CachedConv2d, CachedConvTranspose2d)
from .conv_2d import Conv2d as _Conv2d
from .conv_2d import ConvTranspose2d as _ConvTranspose2d
from .conv_2d import AlignBranches2d as _AlignBranches2d


def chunk_process(f, x, N):
    x = torch.split(x, x.shape[-1] // N, -1)
    y = torch.cat([f(_x) for _x in x], -1)
    return y


def test_equal(model_constructor, input_tensor, crop=True):
    initial_state = USE_CACHED_CONV

    use_cached_conv(False)
    model = model_constructor()
    use_cached_conv(True)
    cmodel = model_constructor()

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    cd = cmodel.cumulative_delay
    if cd > 0:
        y = model(input_tensor)[..., :-cd]
    else:
        y = model(input_tensor)
    cy = chunk_process(lambda x: cmodel(x), input_tensor, 4)[..., cd:]

    if crop and cd > 0:
        y = y[..., cd:-cd]
        cy = cy[..., cd:-cd]

    use_cached_conv(initial_state)
    return torch.allclose(y, cy, 1e-4, 1e-4)


def Conv1d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedConv1d(*args, **kwargs)
    else:
        return _Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedConvTranspose1d(*args, **kwargs)
    else:
        return _ConvTranspose1d(*args, **kwargs)


def AlignBranches1d(*args, **kwargs):
    if USE_CACHED_CONV:
        return _AlignBranches1d(*args, **kwargs)
    else:
        return Branches(*args, **kwargs)


# backward-compatible alias
def AlignBranches(*args, **kwargs):
    return AlignBranches1d(*args, **kwargs)


def Conv2d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedConv2d(*args, **kwargs)
    else:
        return _Conv2d(*args, **kwargs)


def ConvTranspose2d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedConvTranspose2d(*args, **kwargs)
    else:
        return _ConvTranspose2d(*args, **kwargs)


def AlignBranches2d(*args, **kwargs):
    if USE_CACHED_CONV:
        return _AlignBranches2d(*args, **kwargs)
    else:
        return Branches(*args, **kwargs)
