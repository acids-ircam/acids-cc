import pytest
import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)

CHANNELS = 4
DEPTH, HEIGHT = 3, 5
T = 2**10


def make_input(device, batch=1, channels=CHANNELS):
    return torch.randn(batch, channels, DEPTH, HEIGHT, T, device=device)


@pytest.fixture(params=[False, True], ids=["eager", "cached"])
def cached(request):
    cc.use_cached_conv(request.param)
    yield request.param
    cc.use_cached_conv(False)


# (kernel_size, stride, dilation); stride / dilation act on (depth, height,
# time) and only time is streamed
conv_params = [
    (3, 1, 1),
    (5, 1, 1),
    ((1, 3, 5), 1, 1),
    ((3, 1, 7), 1, 1),
    (3, (1, 1, 2), 1),
    (5, (1, 1, 4), 1),
    (3, (1, 2, 2), 1),
    (3, 1, (1, 1, 4)),
    (3, (1, 1, 2), (1, 2, 2)),
]


@pytest.mark.parametrize("params", conv_params)
@pytest.mark.parametrize("padding", ["same", "causal", "anticausal"])
def test_conv_3d(device, params, padding):
    k, s, d = params
    model_constructor = lambda: cc.Conv3d(CHANNELS, CHANNELS, k, stride=s,
                                          dilation=d, padding=padding)
    assert cc.test_equal(model_constructor, make_input(device))


@pytest.mark.parametrize("groups", [2, 4])
def test_conv_3d_groups(device, groups):
    model_constructor = lambda: cc.Conv3d(CHANNELS, CHANNELS, 3,
                                          groups=groups)
    assert cc.test_equal(model_constructor, make_input(device))


# kernel_size / stride / padding as (depth, height, time); kernel_size along
# time must be stride + 2 * padding for the cached transposed convolution
conv_t_params = [
    ((1, 1, 4), (1, 1, 2), (0, 0, 1)),
    ((3, 3, 4), (1, 1, 2), (1, 1, 1)),
    ((1, 3, 8), (1, 1, 4), (0, 1, 2)),
    ((1, 1, 12), (1, 1, 4), (0, 0, 4)),
]


@pytest.mark.parametrize("params", conv_t_params)
def test_conv_t_3d(device, params):
    k, s, p = params
    model_constructor = lambda: cc.ConvTranspose3d(
        CHANNELS, CHANNELS, k, stride=s, padding=p)
    assert cc.test_equal(model_constructor, make_input(device))


def build_stack(kernels, strides, padding):
    layers, cum_delay = [], 0
    for k, s in zip(kernels, strides):
        layers.append(
            cc.Conv3d(CHANNELS, CHANNELS, k, stride=(1, 1, s),
                      padding=padding, cumulative_delay=cum_delay))
        cum_delay = layers[-1].cumulative_delay
    return cc.CachedSequential(*layers)


@pytest.mark.parametrize("strides", [[1, 1, 1], [2, 1, 1], [1, 2, 1],
                                     [2, 1, 2]])
@pytest.mark.parametrize("padding", ["same", "causal"])
def test_sequential_3d(device, strides, padding):
    model_constructor = lambda: build_stack([3, 5, 7], strides, padding)
    assert cc.test_equal(model_constructor, make_input(device))


def test_encoder_decoder_3d(device):
    """strided conv followed by a transposed conv, back to the input rate"""

    def build():
        down = cc.Conv3d(CHANNELS, CHANNELS, 3, stride=(1, 1, 2))
        up = cc.ConvTranspose3d(CHANNELS, CHANNELS, (3, 3, 4),
                                stride=(1, 1, 2), padding=(1, 1, 1),
                                cumulative_delay=down.cumulative_delay)
        return cc.CachedSequential(down, up)

    x = make_input(device)
    assert cc.test_equal(build, x)
    cc.use_cached_conv(False)
    assert build().to(device)(x).shape == x.shape


@pytest.mark.parametrize("params", conv_params)
@pytest.mark.parametrize("padding", ["same", "centered", "causal",
                                     "anticausal"])
def test_implicit_padding_3d(device, cached, params, padding):
    """matches nn.Conv3d on an explicitly padded input"""
    k, s, d = params
    model = cc.Conv3d(CHANNELS, CHANNELS, k, stride=s, dilation=d,
                      padding=padding).to(device)
    reference = nn.Conv3d(CHANNELS, CHANNELS, k, stride=s,
                          dilation=d).to(device)
    reference.load_state_dict(model.state_dict())

    x = make_input(device)
    pad = cc.conv_3d._resolve_3d_padding(model, padding)
    if cached:
        st = model.stride[2]
        delay = (st - pad[1] % st) % st
        assert model.cumulative_delay == (pad[1] + delay) // st
        if delay:
            x_reference = nn.functional.pad(x, (delay, 0))[..., :-delay]
        else:
            x_reference = x
        pad = [pad[0] + pad[1], 0] + pad[2:]
    else:
        assert model.cumulative_delay == 0
        x_reference = x
    y = reference(nn.functional.pad(x_reference, pad))
    out = model(x)  # call once: the cached model is stateful
    assert out.shape == y.shape
    assert torch.allclose(out, y, 1e-4, 1e-4)


def test_same_padding_keeps_shape(device, cached):
    model = cc.Conv3d(CHANNELS, CHANNELS, (3, 4, 5)).to(device)
    x = make_input(device)
    assert model(x).shape == x.shape


def test_unknown_padding_mode_3d(cached):
    with pytest.raises(ValueError):
        cc.Conv3d(CHANNELS, CHANNELS, 3, padding="acausal")


@pytest.mark.parametrize("params", conv_params)
def test_causality_3d(device, cached, params):
    k, s, d = params
    model = cc.Conv3d(CHANNELS, CHANNELS, k, stride=s, dilation=d,
                      padding="causal").to(device)
    other = cc.Conv3d(CHANNELS, CHANNELS, k, stride=s, dilation=d,
                      padding="causal").to(device)
    other.load_state_dict(model.state_dict())
    x = make_input(device)
    changed = x.clone()
    cutoff = 513
    changed[..., cutoff:] += 10

    if cached:
        y = cc.chunk_process(model, x, 4)
        changed_y = cc.chunk_process(other, changed, 4)
    else:
        y, changed_y = model(x), other(changed)
    st = model.stride[2]
    prefix = (cutoff + st - 1) // st
    assert model.cumulative_delay == 0
    assert torch.allclose(y[..., :prefix], changed_y[..., :prefix])
    assert not torch.allclose(y[..., prefix:], changed_y[..., prefix:])


@pytest.mark.parametrize("padding", ["same", "causal"])
def test_residual_3d(device, padding):
    cc.use_cached_conv(False)
    conv = cc.Conv3d(CHANNELS, CHANNELS, 3, padding=padding)
    model = cc.AlignBranches3d(conv, nn.Identity(),
                               delays=[conv.cumulative_delay, 0]).to(device)

    cc.use_cached_conv(True)
    cconv = cc.Conv3d(CHANNELS, CHANNELS, 3, padding=padding)
    cmodel = cc.AlignBranches3d(cconv, nn.Identity(),
                                delays=[cconv.cumulative_delay, 0]).to(device)
    cc.use_cached_conv(False)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = make_input(device)
    cd = cmodel.cumulative_delay
    y = sum(model(x))
    if cd:
        y = y[..., :-cd]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cd:]
    assert torch.allclose(y, cy, 1e-4, 1e-4)


@pytest.mark.parametrize("history_size", ["automatic", 0, 64])
def test_group_norm_3d(device, history_size):
    model = cc.CachedGroupNorm3d(2, CHANNELS, history_size=history_size,
                                 max_batch_size=2).to(device)
    reference = nn.GroupNorm(2, CHANNELS).to(device)
    x = make_input(device, batch=2)
    chunks = x.chunk(4, -1)
    size = chunks[0].shape[-1] if history_size == "automatic" else history_size
    history = x.new_zeros(*x.shape[:-1], size)
    for chunk in chunks:
        window = torch.cat([history, chunk], -1)
        expected = reference(window)[..., -chunk.shape[-1]:]
        assert torch.allclose(model(chunk), expected, 1e-4, 1e-4)
        if size:
            history = window[..., -size:]


@pytest.mark.parametrize("build", [
    lambda: cc.CachedConv3d(CHANNELS, CHANNELS, 3),
    lambda: cc.CachedConv3d(CHANNELS, CHANNELS, 3, stride=(1, 1, 2),
                            padding="causal"),
    lambda: cc.CachedConvTranspose3d(CHANNELS, CHANNELS, (1, 1, 4),
                                     stride=(1, 1, 2), padding=(0, 0, 1)),
], ids=["conv", "conv_strided_causal", "conv_t"])
def test_script_3d(device, build):
    model = build().to(device)
    x = make_input(device)
    y = model(x)  # caches are allocated on the first call
    scripted = torch.jit.script(model)
    assert scripted(x).shape == y.shape
