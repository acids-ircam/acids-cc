import pytest
import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)

CHANNELS = 16
HEIGHT = 8

# (kernel_size, stride, dilation)
conv_params = [(3, 1, 1), (5, 1, 1), (3, 2, 1), (7, 2, 1), (7, 4, 1),
               (3, 1, 4), (5, 2, 2)]

modes = ["same", "causal", "anticausal"]


@pytest.fixture(params=[False, True], ids=["eager", "cached"])
def cached(request):
    cc.use_cached_conv(request.param)
    yield request.param
    cc.use_cached_conv(False)


@pytest.mark.parametrize("params", conv_params)
@pytest.mark.parametrize("mode", modes)
def test_implicit_padding_1d(cached, params, mode):
    """padding="same" builds the same layer as padding=cc.get_padding(...)"""
    k, s, d = params
    explicit_mode = "centered" if mode == "same" else mode

    implicit = cc.Conv1d(CHANNELS, CHANNELS, k, stride=s, dilation=d,
                         padding=mode)
    explicit = cc.Conv1d(CHANNELS, CHANNELS, k, stride=s, dilation=d,
                         padding=cc.get_padding(k, s, d, explicit_mode))

    for p1, p2 in zip(implicit.parameters(), explicit.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, CHANNELS, 2**12)
    assert implicit.cumulative_delay == explicit.cumulative_delay
    assert torch.allclose(implicit(x), explicit(x))


@pytest.mark.parametrize("params", conv_params)
@pytest.mark.parametrize("mode", modes)
def test_implicit_padding_2d(cached, params, mode):
    k, s, d = params
    explicit_mode = "centered" if mode == "same" else mode

    implicit = cc.Conv2d(CHANNELS, CHANNELS, k, stride=(1, s), dilation=(1, d),
                         padding_time=mode, padding_vert="same")
    explicit = cc.Conv2d(CHANNELS, CHANNELS, k, stride=(1, s), dilation=(1, d),
                         padding_time=cc.get_padding(k, s, d, explicit_mode),
                         padding_vert="same")

    for p1, p2 in zip(implicit.parameters(), explicit.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, CHANNELS, HEIGHT, 2**12)
    assert implicit.cumulative_delay == explicit.cumulative_delay
    assert torch.allclose(implicit(x), explicit(x))


@pytest.mark.parametrize("params", conv_params)
def test_streaming_implicit_padding_1d(params):
    k, s, d = params
    model_constructor = lambda: cc.Conv1d(
        CHANNELS, CHANNELS, k, stride=s, dilation=d, padding="same")
    input_tensor = torch.randn(1, CHANNELS, 2**14)
    assert cc.test_equal(model_constructor, input_tensor)


@pytest.mark.parametrize("params", conv_params)
def test_streaming_implicit_padding_2d(params):
    k, s, d = params
    model_constructor = lambda: cc.Conv2d(CHANNELS,
                                          CHANNELS,
                                          k,
                                          stride=(1, s),
                                          dilation=(1, d),
                                          padding_time="same",
                                          padding_vert="same")
    input_tensor = torch.randn(1, CHANNELS, HEIGHT, 2**14)
    assert cc.test_equal(model_constructor, input_tensor)


def test_padding_shorthand_2d(cached):
    """padding= applies to both axes, and is not silently ignored"""
    model = cc.Conv2d(CHANNELS, CHANNELS, 3, padding="same")
    x = torch.randn(1, CHANNELS, HEIGHT, 2**10)
    assert model(x).shape == x.shape


def test_padding_positional(cached):
    """padding given positionally is resolved as well"""
    model = cc.Conv1d(CHANNELS, CHANNELS, 5, 1, "same")
    x = torch.randn(1, CHANNELS, 2**10)
    assert model(x).shape == x.shape


@pytest.mark.parametrize("padding", ["valid", "none", 0, None])
def test_no_padding(cached, padding):
    model = cc.Conv1d(CHANNELS, CHANNELS, 5, padding=padding)
    x = torch.randn(1, CHANNELS, 2**10)
    assert model(x).shape[-1] == x.shape[-1] - 4
    assert model.cumulative_delay == 0


def test_int_padding(cached):
    """an int pads both sides, as in torch"""
    model = cc.Conv1d(CHANNELS, CHANNELS, 5, padding=2)
    x = torch.randn(1, CHANNELS, 2**10)
    assert model(x).shape == x.shape


def test_unknown_padding_mode(cached):
    with pytest.raises(Exception):
        cc.Conv1d(CHANNELS, CHANNELS, 5, padding="acausal")


def test_streaming_sequential_implicit():
    """a stack built without any explicit padding streams correctly"""

    def build_model():
        layers = []
        cum_delay = 0
        for k, s in zip([3, 5, 7, 9], [2, 1, 2, 1]):
            layers.append(
                cc.Conv1d(CHANNELS,
                          CHANNELS,
                          k,
                          stride=s,
                          padding="same",
                          cumulative_delay=cum_delay))
            cum_delay = layers[-1].cumulative_delay
        return cc.CachedSequential(*layers)

    assert cc.test_equal(build_model, torch.randn(1, CHANNELS, 2**14))
