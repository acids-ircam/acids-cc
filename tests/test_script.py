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
        "padding": "same",
    }, (1, 16, 2**14)),
    (cc.CachedConvTranspose1d, {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
    }, (1, 16, 2**14)),
]

# Cached 2D convolutions are scriptable after cache initialisation.
model_list_2d = [
    (cc.CachedConv2d, {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 3,
        "padding": padding,
    }, (1, 16, 8, 2**14))
    for padding in ["same", "causal"]
] + [
    (cc.CachedConvTranspose2d, {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": (1, 4),
        "stride": (1, 2),
        "padding": (0, 1),
    }, (1, 16, 8, 2**14)),
]


@pytest.mark.parametrize("model_spec", model_list_1d)
def test_script_1d(device, model_spec):
    cls, hp, shape = model_spec
    model = cls(**hp).to(device)
    x = torch.randn(*shape, device=device)
    model(x)
    torch.jit.script(model)


@pytest.mark.parametrize("model_spec", model_list_2d)
def test_script_2d(device, model_spec):
    cls, hp, shape = model_spec
    cc.use_cached_conv(True)
    try:
        model = cls(**hp).to(device)
        x = torch.randn(*shape, device=device)
        model(x)
        torch.jit.script(model)
    finally:
        cc.use_cached_conv(False)
