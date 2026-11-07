import pytest
import torch
import cached_conv as cc

torch.set_grad_enabled(False)

hparams_list_1d = [
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1
    },
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 8,
        "stride": 4,
        "padding": 2
    },
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 12,
        "stride": 4,
        "padding": 4
    },
]

# kernel_size/stride/padding as (height, time) tuples; height dim is kept at 1
hparams_list_2d = [
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": (1, 4),
        "stride": (1, 2),
        "padding": (0, 1),
    },
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": (1, 8),
        "stride": (1, 4),
        "padding": (0, 2),
    },
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": (1, 12),
        "stride": (1, 4),
        "padding": (0, 4),
    },
]


@pytest.mark.parametrize("hparams", hparams_list_1d)
def test_conv_t_1d(device, hparams):
    model_constructor = lambda: cc.ConvTranspose1d(**hparams)
    input_tensor = torch.randn(1, hparams["out_channels"], 2**14, device=device)
    assert cc.test_equal(model_constructor, input_tensor)


@pytest.mark.parametrize("hparams", hparams_list_2d)
def test_conv_t_2d(device, hparams):
    model_constructor = lambda: cc.ConvTranspose2d(**hparams)
    input_tensor = torch.randn(1, hparams["out_channels"], 8, 2**14, device=device)
    assert cc.test_equal(model_constructor, input_tensor)
