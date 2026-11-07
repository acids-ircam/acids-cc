import pytest
import torch

import cached_conv as cc

torch.set_grad_enabled(False)

hparams_list_1d = [{
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "padding": "same"
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 5,
    "padding": "same"
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "stride": 2,
    "padding": "same"
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 2,
    "padding": "same"
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 4,
    "padding": "same"
}]

hparams_list_2d = [{
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 5,
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "stride": 2,
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 2,
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 4,
}]


@pytest.mark.parametrize("hparams", hparams_list_1d)
def test_conv_1d(device, hparams):
    model_constructor = lambda: cc.Conv1d(**hparams)
    input_tensor = torch.randn(1, hparams["in_channels"], 2**14, device=device)
    assert cc.test_equal(model_constructor, input_tensor)


@pytest.mark.parametrize("hparams", hparams_list_2d)
@pytest.mark.parametrize("padding", ["same", "causal", "anticausal"])
def test_conv_2d(device, hparams, padding):
    model_constructor = lambda: cc.Conv2d(**hparams, padding=padding)
    input_tensor = torch.randn(1, hparams["in_channels"], 8, 2**14, device=device)
    assert cc.test_equal(model_constructor, input_tensor)
