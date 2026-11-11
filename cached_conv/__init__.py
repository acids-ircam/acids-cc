import torch

USE_CACHED_CONV = False
MAX_BATCH_SIZE = 64 # Warning! added for retro-compatibility

def use_cached_conv(state: bool):
    global USE_CACHED_CONV
    USE_CACHED_CONV = state


def is_cached():
    global USE_CACHED_CONV
    return USE_CACHED_CONV


def __getattr__(name):
    # backward compatibility with cached_conv < 3: ``cc.USE_BUFFER_CONV`` was
    # the name of the global flag. Resolved dynamically so that it always
    # mirrors ``USE_CACHED_CONV`` (set it with ``use_cached_conv``).
    if name == "USE_BUFFER_CONV":
        return USE_CACHED_CONV
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .utils import Branches, normalization
from .conv_1d import (get_padding, CachedSequential, Sequential,
                      CachedPadding, CachedConv1d, CachedConvTranspose1d,
                      CachedGroupNorm, CachedGroupNorm1d)
from .conv_1d import Conv1d as _Conv1d
from .conv_1d import ConvTranspose1d as _ConvTranspose1d
from .conv_1d import AlignBranches1d as _AlignBranches1d
from .conv_2d import CachedConv2d, CachedConvTranspose2d, CachedGroupNorm2d
from .conv_2d import Conv2d as _Conv2d
from .conv_2d import ConvTranspose2d as _ConvTranspose2d
from .conv_2d import AlignBranches2d as _AlignBranches2d
from .conv_3d import CachedConv3d, CachedConvTranspose3d, CachedGroupNorm3d
from .conv_3d import Conv3d as _Conv3d
from .conv_3d import ConvTranspose3d as _ConvTranspose3d
from .conv_3d import AlignBranches3d as _AlignBranches3d
from .variants import (get_window, CachedWindowedConv1d,
                       CachedWindowedConv2d, CachedWindowedConv3d,
                       CachedModulatedConv1d, CachedModulatedConv2d,
                       CachedModulatedConv3d)
from . import variants as _variants
from .upsample import CachedUpsample1d, CachedUpsample2d, CachedUpsample3d
from . import upsample as _upsample


def chunk_process(f, x, N):
    x = torch.split(x, x.shape[-1] // N, -1)
    y = torch.cat([f(_x) for _x in x], -1)
    return y


def test_equal(model_constructor, input_tensor, crop=True):
    initial_state = USE_CACHED_CONV

    use_cached_conv(False)
    model = model_constructor().to(input_tensor.device)
    use_cached_conv(True)
    cmodel = model_constructor().to(input_tensor.device)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    # delay of the cached model w.r.t. the non-cached one (which may itself
    # be delayed, e.g. ConvTranspose2d with crop_right=True)
    cd = cmodel.cumulative_delay - model.cumulative_delay
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


def Conv3d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedConv3d(*args, **kwargs)
    else:
        return _Conv3d(*args, **kwargs)


def ConvTranspose3d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedConvTranspose3d(*args, **kwargs)
    else:
        return _ConvTranspose3d(*args, **kwargs)


def AlignBranches3d(*args, **kwargs):
    if USE_CACHED_CONV:
        return _AlignBranches3d(*args, **kwargs)
    else:
        return Branches(*args, **kwargs)


def WindowedConv1d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedWindowedConv1d(*args, **kwargs)
    else:
        return _variants.WindowedConv1d(*args, **kwargs)


def WindowedConv2d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedWindowedConv2d(*args, **kwargs)
    else:
        return _variants.WindowedConv2d(*args, **kwargs)


def WindowedConv3d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedWindowedConv3d(*args, **kwargs)
    else:
        return _variants.WindowedConv3d(*args, **kwargs)


def ModulatedConv1d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedModulatedConv1d(*args, **kwargs)
    else:
        return _variants.ModulatedConv1d(*args, **kwargs)


def ModulatedConv2d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedModulatedConv2d(*args, **kwargs)
    else:
        return _variants.ModulatedConv2d(*args, **kwargs)


def ModulatedConv3d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedModulatedConv3d(*args, **kwargs)
    else:
        return _variants.ModulatedConv3d(*args, **kwargs)


def Upsample1d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedUpsample1d(*args, **kwargs)
    else:
        return _upsample.Upsample1d(*args, **kwargs)


def Upsample2d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedUpsample2d(*args, **kwargs)
    else:
        return _upsample.Upsample2d(*args, **kwargs)


def Upsample3d(*args, **kwargs):
    if USE_CACHED_CONV:
        return CachedUpsample3d(*args, **kwargs)
    else:
        return _upsample.Upsample3d(*args, **kwargs)
