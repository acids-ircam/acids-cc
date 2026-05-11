import pytest
import torch

import cached_conv as cc

torch.set_grad_enabled(False)

# Each entry: (class, kwargs, input_shape)
# For 2D models that require cached mode, use_cached_conv(True) is called in the fixture.
model_list_1d = [
    (cc.CachedConv1d, {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 3,
        "padding": cc.get_padding(3),
    }, (1, 16, 2**14)),
    (cc.CachedConvTranspose1d, {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
    }, (1, 16, 2**14)),
]

# CachedConvTranspose2d is scriptable after cache initialisation.
# CachedConv2d is not directly scriptable due to Union[CachedPadding2d, Identity] on self.cache.
model_list_2d = [
    (cc.CachedConvTranspose2d, {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": (1, 4),
        "stride": (1, 2),
        "padding": (0, 1),
    }, (1, 16, 8, 2**14)),
]


@pytest.mark.parametrize("model_spec", model_list_1d)
def test_script_1d(model_spec):
    cls, hp, shape = model_spec
    model = cls(**hp)
    x = torch.randn(*shape)
    model(x)
    torch.jit.script(model)


@pytest.mark.parametrize("model_spec", model_list_2d)
def test_script_2d(model_spec):
    cls, hp, shape = model_spec
    cc.use_cached_conv(True)
    try:
        model = cls(**hp)
        x = torch.randn(*shape)
        model(x)
        torch.jit.script(model)
    finally:
        cc.use_cached_conv(False)
