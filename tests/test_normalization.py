"""``normalization`` keyword of the 1d / 2d / 3d convolutions."""
import copy

import pytest
import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)

CHANNELS = 4

# factory, input shape (without time), kernel_size, stride, padding kwargs
LAYERS = {
    "1d": (lambda: cc.Conv1d, (1, CHANNELS), 5, 2, dict(padding="causal")),
    "2d": (lambda: cc.Conv2d, (1, CHANNELS, 6), (3, 5), (1, 2),
           dict(padding="causal")),
    "3d": (lambda: cc.Conv3d, (1, CHANNELS, 2, 6), (1, 3, 5), (1, 1, 2),
           dict(padding="causal")),
}


@pytest.fixture(params=[False, True], ids=["eager", "cached"])
def cached(request):
    initial = cc.is_cached()
    cc.use_cached_conv(request.param)
    yield request.param
    cc.use_cached_conv(initial)


def build(dim, **kwargs):
    factory, shape, k, s, padding = LAYERS[dim]
    layer = factory()(CHANNELS, CHANNELS, k, stride=s, **padding, **kwargs)
    return layer, torch.randn(*shape, 64)


@pytest.mark.parametrize("dim", LAYERS)
@pytest.mark.parametrize("mode", [None, False, "identity"])
def test_no_normalization(cached, dim, mode):
    layer, _ = build(dim, normalization=mode)
    names = dict(layer.named_parameters())
    assert "weight" in names and "weight_g" not in names


@pytest.mark.parametrize("dim", LAYERS)
@pytest.mark.parametrize("mode", [True, "weight_norm"])
def test_weight_norm(cached, dim, mode):
    """Same output as the unnormalized layer with weight = g * v / |v|."""
    layer, x = build(dim, normalization=mode)
    names = dict(layer.named_parameters())
    assert {"weight_g", "weight_v"} <= names.keys() and "weight" not in names

    # rescale g so that the normalization actually matters
    layer.weight_g.data.mul_(torch.rand_like(layer.weight_g) + .5)
    reference, _ = build(dim)
    reference.bias.data.copy_(layer.bias.data)
    v = layer.weight_v
    norm = v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))
    reference.weight.data.copy_(layer.weight_g * v / norm)

    assert torch.allclose(layer(x), reference(x), 1e-5, 1e-5)


@pytest.mark.parametrize("dim", LAYERS)
def test_unknown_normalization(cached, dim):
    with pytest.raises(ValueError):
        build(dim, normalization="spectral")


@pytest.mark.parametrize("dim", LAYERS)
def test_weight_norm_streaming(dim):
    factory, shape, k, s, padding = LAYERS[dim]
    model_constructor = lambda: factory()(CHANNELS, CHANNELS, k, stride=s,
                                          normalization=True, **padding)
    assert cc.test_equal(model_constructor, torch.randn(*shape, 2**10))


@pytest.mark.parametrize("dim", LAYERS)
def test_weight_norm_scriptable(cached, dim):
    layer, x = build(dim, normalization=True)
    twin = copy.deepcopy(layer)
    # cached layers allocate their cache on the first call (zeros: no history)
    layer(torch.zeros_like(x))
    twin(torch.zeros_like(x))
    scripted = torch.jit.script(layer)
    assert "weight_g" not in dict(scripted.named_parameters())
    assert torch.allclose(scripted(x), twin(x), 1e-5, 1e-5)
