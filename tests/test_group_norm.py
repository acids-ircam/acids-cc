import pytest
import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)

hparams_list = [{
    "num_groups": 1,
    "num_channels": 16,
}, {
    "num_groups": 4,
    "num_channels": 16,
}, {
    "num_groups": 16,
    "num_channels": 16,
    "affine": False,
}]


@pytest.mark.parametrize("hparams", hparams_list)
@pytest.mark.parametrize("shape", [(2, 16, 256), (2, 16, 8, 256)])
@pytest.mark.parametrize("history_size", ["automatic", 0, 32, 128])
def test_group_norm(hparams, shape, history_size):
    model = cc.CachedGroupNorm(**hparams, history_size=history_size,
                               max_batch_size=shape[0])
    reference = nn.GroupNorm(**hparams)
    if model.affine:
        model.weight.copy_(torch.randn_like(model.weight))
        model.bias.copy_(torch.randn_like(model.bias))
    reference.load_state_dict(model.state_dict())

    x = torch.randn(*shape)
    chunks = x.chunk(4, dim=-1)
    size = chunks[0].shape[-1] if history_size == "automatic" else history_size
    history = x.new_zeros(*shape[:-1], size)
    for chunk in chunks:
        window = torch.cat([history, chunk], dim=-1)
        expected = reference(window)[..., -chunk.shape[-1]:]
        assert torch.allclose(model(chunk), expected, 1e-4, 1e-4)
        if size:
            history = window[..., -size:]

    model.reset_cache()
    window = torch.cat([torch.zeros_like(history), chunks[0]], dim=-1)
    expected = reference(window)[..., -chunks[0].shape[-1]:]
    assert torch.allclose(model(chunks[0]), expected, 1e-4, 1e-4)
