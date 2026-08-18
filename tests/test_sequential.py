import pytest
import torch

import cached_conv as cc

torch.set_grad_enabled(False)

hparams_list = [{
    "kernels": [3, 5, 7, 9],
    "strides": [1, 1, 1, 1]
}, {
    "kernels": [3, 5, 7, 9],
    "strides": [2, 1, 1, 1]
}, {
    "kernels": [3, 5, 7, 9],
    "strides": [1, 2, 1, 1]
}, {
    "kernels": [3, 5, 7, 9],
    "strides": [1, 1, 2, 1]
}, {
    "kernels": [3, 3, 3, 3],
    "strides": [2, 1, 2, 1]
}]

CHANNELS = 16
HEIGHT = 8


def build_model_1d(kernels, strides):
    layers = []
    cum_delay = 0
    for k, s in zip(kernels, strides):
        layers.append(
            cc.Conv1d(
                CHANNELS, CHANNELS, k,
                stride=s,
                padding="same",
                cumulative_delay=cum_delay,
            ))
        cum_delay = layers[-1].cumulative_delay
    return cc.CachedSequential(*layers)


def build_model_2d(kernels, strides):
    layers = []
    cum_delay = 0
    for k, s in zip(kernels, strides):
        layers.append(
            cc.Conv2d(
                CHANNELS, CHANNELS, k,
                stride=(1, s),
                padding_time="same",
                padding_vert="same",
                cumulative_delay=cum_delay,
            ))
        cum_delay = layers[-1].cumulative_delay
    return cc.CachedSequential(*layers)


@pytest.mark.parametrize("hparams", hparams_list)
def test_sequential_1d(hparams):
    model_constructor = lambda: build_model_1d(**hparams)
    input_tensor = torch.randn(1, CHANNELS, 2**14)
    assert cc.test_equal(model_constructor, input_tensor)


@pytest.mark.parametrize("hparams", hparams_list)
def test_sequential_2d(hparams):
    model_constructor = lambda: build_model_2d(**hparams)
    input_tensor = torch.randn(1, CHANNELS, HEIGHT, 2**14)
    assert cc.test_equal(model_constructor, input_tensor)
