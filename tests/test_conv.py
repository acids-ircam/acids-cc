import pytest
import torch

import cached_conv as cc

torch.set_grad_enabled(False)

hparams_list_1d = [{
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "padding": cc.get_padding(3)
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 5,
    "padding": cc.get_padding(5)
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "stride": 2,
    "padding": cc.get_padding(3, 2)
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 2,
    "padding": cc.get_padding(7, 2)
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 4,
    "padding": cc.get_padding(7, 4)
}]

hparams_list_2d = [{
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "padding_time": cc.get_padding(3),
    "padding_vert": "same",
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 5,
    "padding_time": cc.get_padding(5),
    "padding_vert": "same",
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "stride": 2,
    "padding_time": cc.get_padding(3, 2),
    "padding_vert": "same",
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 2,
    "padding_time": cc.get_padding(7, 2),
    "padding_vert": "same",
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 4,
    "padding_time": cc.get_padding(7, 4),
    "padding_vert": "same",
}]


@pytest.mark.parametrize("hparams", hparams_list_1d)
def test_conv_1d(hparams):
    model_constructor = lambda: cc.Conv1d(**hparams)
    input_tensor = torch.randn(1, hparams["in_channels"], 2**14)
    assert cc.test_equal(model_constructor, input_tensor)


@pytest.mark.parametrize("hparams", hparams_list_2d)
def test_conv_2d(hparams):
    model_constructor = lambda: cc.Conv2d(**hparams)
    input_tensor = torch.randn(1, hparams["in_channels"], 8, 2**14)
    assert cc.test_equal(model_constructor, input_tensor)
