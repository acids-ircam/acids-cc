"""Checks that the 2d layers of cached_conv reproduce RAVE's legacy
``rave/cachedConv2D.py`` implementation (padding_time / padding_vert,
weight normalization, cached streaming).

The legacy file is looked up at $RAVE_CACHED_CONV_2D, or in a RAVE checkout
next to this repository; tests are skipped if it cannot be found.
"""
import copy
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)

LEGACY_PATH = Path(
    os.environ.get(
        "RAVE_CACHED_CONV_2D",
        Path(__file__).parents[2] / "RAVE" / "rave" / "cachedConv2D.py"))

pytestmark = pytest.mark.skipif(not LEGACY_PATH.exists(),
                                reason=f"{LEGACY_PATH} not found")

IN_CH, OUT_CH, HEIGHT = 4, 6, 12
N_CHUNKS = 8


@pytest.fixture(scope="module")
def legacy():
    spec = importlib.util.spec_from_file_location("rave_cached_conv_2d",
                                                  LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=[False, True], ids=["eager", "cached"])
def cached(request):
    """use_cached_conv also drives the legacy USE_BUFFER_CONV alias."""
    initial = cc.is_cached()
    cc.use_cached_conv(request.param)
    yield request.param
    cc.use_cached_conv(initial)


def test_use_buffer_conv_alias():
    initial = cc.is_cached()
    for state in (True, False):
        cc.use_cached_conv(state)
        assert cc.USE_BUFFER_CONV is state
    cc.use_cached_conv(initial)
    with pytest.raises(AttributeError):
        cc.SOME_UNKNOWN_ATTRIBUTE


def copy_params(src, dst):
    """Copies parameters by name, ignoring the legacy ``causal_conv.`` prefix."""
    src_params = {
        k.replace("causal_conv.", ""): v
        for k, v in src.named_parameters()
    }
    dst_params = dict(dst.named_parameters())
    assert src_params.keys() == dst_params.keys()
    for k, v in dst_params.items():
        v.data.copy_(src_params[k].data)


def run(model, x, cached):
    """Whole tensor in eager mode, N_CHUNKS successive chunks in cached mode."""
    if not cached:
        return model(x)
    return torch.cat([model(c) for c in x.chunk(N_CHUNKS, -1)], -1)


# kernel_size, stride, dilation
conv_params = [
    (3, 1, 1),
    (5, 1, 1),
    ((3, 5), 1, 1),
    (3, (1, 2), 1),
    ((5, 5), (2, 2), 1),
    (3, 1, (1, 2)),
    ((5, 3), (2, 4), (1, 3)),
    (1, 1, 1),
]


def kernel_axis(value, axis):
    return value[axis] if isinstance(value, tuple) else value


paddings_time = {
    "causal": lambda k, s, d: cc.get_padding(k, s, d, mode="causal"),
    "centered": lambda k, s, d: cc.get_padding(k, s, d, mode="centered"),
    "anticausal": lambda k, s, d: cc.get_padding(k, s, d, mode="anticausal"),
    "int": lambda k, s, d: (k - 1) // 2,
    "zero": lambda k, s, d: 0,
}

paddings_vert = {
    "same": lambda k: "same",
    "none": lambda k: "none",
    "int": lambda k: (k - 1) // 2,
    "tuple": lambda k: (k - 1, 0),
}


@pytest.mark.parametrize("normalization", [False, True], ids=["raw", "wn"])
@pytest.mark.parametrize("pad_vert", paddings_vert.keys())
@pytest.mark.parametrize("pad_time", paddings_time.keys())
@pytest.mark.parametrize("params", conv_params, ids=str)
def test_conv2d_matches_legacy(legacy, cached, params, pad_time, pad_vert,
                               normalization):
    k, s, d = params
    kt, st, dt = (kernel_axis(v, 1) for v in params)
    kwargs = dict(kernel_size=k,
                  stride=s,
                  dilation=d,
                  padding_time=paddings_time[pad_time](kt, st, dt),
                  padding_vert=paddings_vert[pad_vert](kernel_axis(k, 0)))

    reference = legacy.CachedConv2d(IN_CH, OUT_CH, normalization=normalization,
                                    **kwargs)
    model = cc.Conv2d(IN_CH, OUT_CH, normalization=normalization, **kwargs)
    copy_params(reference, model)

    assert model.cumulative_delay == reference.cumulative_delay

    x = torch.randn(2, IN_CH, HEIGHT, 64 * N_CHUNKS)
    y_ref = run(reference, x, cached)
    y = run(model, x, cached)
    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, 1e-5, 1e-5)


def test_padding_is_padding_time(cached):
    x = torch.randn(1, IN_CH, HEIGHT, 128)
    a = cc.Conv2d(IN_CH, OUT_CH, 5, padding=(4, 0), padding_vert=1)
    b = cc.Conv2d(IN_CH, OUT_CH, 5, padding_time=(4, 0), padding_vert=1)
    b.load_state_dict(a.state_dict())
    assert a.cumulative_delay == b.cumulative_delay
    assert torch.allclose(a(x), b(x))


def test_padding_and_padding_time_exclusive(cached):
    with pytest.raises(ValueError):
        cc.Conv2d(IN_CH, OUT_CH, 3, padding="causal", padding_time="causal")
    with pytest.raises(ValueError):
        cc.Conv2d(IN_CH, OUT_CH, 3, padding=1, padding_time=1)


def test_padding_vert_modes(cached):
    x = torch.randn(1, IN_CH, HEIGHT, 128)
    assert cc.Conv2d(IN_CH, OUT_CH, 5)(x).shape[-2] == HEIGHT
    assert cc.Conv2d(IN_CH, OUT_CH, 5, padding_vert="none")(x).shape[-2] == HEIGHT - 4
    assert cc.Conv2d(IN_CH, OUT_CH, 5, padding_vert=(4, 0))(x).shape[-2] == HEIGHT
    assert cc.Conv2d(IN_CH, OUT_CH, 5, padding_vert=3)(x).shape[-2] == HEIGHT + 2
    with pytest.raises(ValueError):
        cc.Conv2d(IN_CH, OUT_CH, 3, padding_vert="acausal")


@pytest.mark.parametrize("params", [(4, 2, 1), (8, 4, 2), (3, 1, 1), (6, 2, 2)],
                         ids=str)
def test_conv_transpose2d_matches_legacy(legacy, cached, params):
    """With crop_right=True (default), the non-cached layer crops the end of
    the output like the legacy one, so both modes match it exactly."""
    k, s, p = params
    kwargs = dict(kernel_size=(3, k), stride=(1, s), padding=(1, p))
    reference = legacy.CachedConvTranspose2d(IN_CH, OUT_CH, **kwargs)
    model = cc.ConvTranspose2d(IN_CH, OUT_CH, **kwargs)
    copy_params(reference, model)
    if cached:
        assert model.cumulative_delay == reference.cumulative_delay
    else:
        # the legacy layer reports 0 although its output is delayed
        assert model.cumulative_delay == p

    x = torch.randn(2, IN_CH, HEIGHT, 32 * N_CHUNKS)
    y, y_ref = run(model, x, cached), run(reference, x, cached)
    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, 1e-5, 1e-5)


@pytest.mark.parametrize("params", [(4, 2, 1), (8, 4, 2), (6, 2, 2)], ids=str)
def test_conv_transpose2d_crop_right(params):
    k, s, p = params
    kwargs = dict(kernel_size=(3, k), stride=(1, s), padding=(1, p))
    torch_layer = nn.ConvTranspose2d(IN_CH, OUT_CH, **kwargs)
    centered = cc.ConvTranspose2d(IN_CH, OUT_CH, crop_right=False, **kwargs)
    right = cc.ConvTranspose2d(IN_CH, OUT_CH, **kwargs)
    centered.load_state_dict(torch_layer.state_dict())
    right.load_state_dict(torch_layer.state_dict())

    x = torch.randn(2, IN_CH, HEIGHT, 64)
    y_torch, y_centered, y_right = torch_layer(x), centered(x), right(x)
    # crop_right=False is torch's behavior
    assert centered.cumulative_delay == 0
    assert torch.allclose(y_centered, y_torch, 1e-5, 1e-5)
    # crop_right=True: same length, delayed by padding[1]
    assert right.cumulative_delay == p
    assert y_right.shape == y_torch.shape
    assert torch.allclose(y_right[..., p:], y_torch[..., :-p], 1e-5, 1e-5)
    # delay accumulates with the upstream one, as in the cached layer
    stacked = cc.ConvTranspose2d(IN_CH, OUT_CH, cumulative_delay=3, **kwargs)
    assert stacked.cumulative_delay == p + 3 * s


@pytest.mark.parametrize("crop_right", [True, False])
def test_conv_transpose2d_is_scriptable(crop_right):
    model = cc.ConvTranspose2d(IN_CH, OUT_CH, (3, 4), stride=(1, 2),
                               padding=(1, 1), crop_right=crop_right)
    x = torch.randn(1, IN_CH, HEIGHT, 32)
    assert torch.allclose(torch.jit.script(model)(x), model(x))


def test_align_branches2d_matches_legacy(legacy, cached):
    def build(conv, **extra):
        return [
            conv(IN_CH, OUT_CH, k, padding_time=cc.get_padding(k), padding_vert="same", **extra)
            for k in (1, 3, 7)
        ]

    ref_branches = build(legacy.CachedConv2d, normalization=False)
    branches_ = build(cc.Conv2d)
    for r, b in zip(ref_branches, branches_):
        copy_params(r, b)

    reference = legacy.AlignBranches2d(*ref_branches)
    model = cc.AlignBranches2d(*branches_)
    assert model.cumulative_delay == reference.cumulative_delay

    x = torch.randn(2, IN_CH, HEIGHT, 64 * N_CHUNKS)
    if cached:
        outs_ref = [reference(c) for c in x.chunk(N_CHUNKS, -1)]
        outs = [model(c) for c in x.chunk(N_CHUNKS, -1)]
        outs_ref = [torch.cat(o, -1) for o in zip(*outs_ref)]
        outs = [torch.cat(o, -1) for o in zip(*outs)]
    else:
        outs_ref, outs = reference(x), model(x)
    for y, y_ref in zip(outs, outs_ref):
        assert torch.allclose(y, y_ref, 1e-5, 1e-5)


# ----- normalization ---------------------------------------------------------


def test_normalization_modes():
    for mode in (None, False, "identity"):
        conv = nn.Conv2d(2, 2, 3)
        assert cc.normalization(conv, mode) is conv
        assert "weight" in dict(conv.named_parameters())
    for mode in (True, "weight_norm"):
        conv = cc.normalization(nn.Conv2d(2, 2, 3), mode)
        names = dict(conv.named_parameters())
        assert "weight_g" in names and "weight_v" in names
    with pytest.raises(ValueError):
        cc.normalization(nn.Conv2d(2, 2, 3), "spectral")


def test_normalized_conv2d_is_scriptable(cached):
    model = cc.Conv2d(IN_CH, OUT_CH, (3, 5), stride=(1, 2),
                      padding_time="causal", padding_vert="same",
                      normalization=True)
    twin = copy.deepcopy(model)
    x = torch.randn(1, IN_CH, HEIGHT, 128)
    # cached layers allocate their cache on the first call (zeros: no history)
    model(torch.zeros_like(x))
    twin(torch.zeros_like(x))
    scripted = torch.jit.script(model)
    assert "weight_g" not in dict(scripted.named_parameters())
    assert torch.allclose(scripted(x), twin(x), 1e-5, 1e-5)
