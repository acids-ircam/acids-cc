"""Windowed convolutions, modulated convolutions and cached upsampling."""
import pytest
import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)

C = 4
T = 2**10
SPATIAL = {1: (), 2: (6, ), 3: (3, 5)}


def make_input(device, dims, batch=1, channels=C):
    return torch.randn(batch, channels, *SPATIAL[dims], T, device=device)


def factory(kind, dims):
    return getattr(cc, f"{kind}{dims}d")


def conv_kwargs(dims, stride=1, padding="same", dilation=1):
    """time stride / dilation, as accepted by the 1d, 2d and 3d layers"""
    if dims == 1:
        return dict(stride=stride, padding=padding, dilation=dilation)
    lead = (1, ) * (dims - 1)
    return dict(stride=lead + (stride, ), padding=padding,
                dilation=lead + (dilation, ))


@pytest.fixture(params=[False, True], ids=["eager", "cached"])
def cached(request):
    cc.use_cached_conv(request.param)
    yield request.param
    cc.use_cached_conv(False)


# ----- windows ---------------------------------------------------------------


def test_get_window():
    assert torch.allclose(cc.get_window("hann", 3), torch.tensor([.5, 1, .5]))
    assert torch.allclose(cc.get_window("hann", 1), torch.ones(1))
    assert (cc.get_window("hann", 9) > 0).all(), "no tap should be zeroed"
    assert torch.equal(cc.get_window(None, 4), torch.ones(4))
    assert torch.equal(cc.get_window("rect", 4), torch.ones(4))
    assert torch.equal(cc.get_window([1, 2, 3], 3), torch.tensor([1., 2, 3]))
    assert torch.equal(cc.get_window(torch.ones, 5), torch.ones(5))
    with pytest.raises(ValueError):
        cc.get_window("triangle", 3)
    with pytest.raises(ValueError):
        cc.get_window([1, 2], 3)


# ----- windowed convolutions -------------------------------------------------


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("window", ["hann", "blackman", "kaiser", None])
def test_windowed_matches_reference(device, cached, dims, window):
    """same as a plain conv whose weight was multiplied by the window"""
    kw = conv_kwargs(dims)
    model = factory("WindowedConv", dims)(C, C, 5, window=window,
                                          **kw).to(device)
    reference = factory("Conv", dims)(C, C, 5, **kw).to(device)

    full = torch.ones(())
    for axis in range(dims):
        shape = [1] * dims
        shape[axis] = 5
        full = full * cc.get_window(window, 5).view(shape)
    reference.weight.copy_(model.weight * full.to(device))
    reference.bias.copy_(model.bias)

    x = make_input(device, dims)
    assert torch.allclose(model(x), reference(x), 1e-5, 1e-5)


def test_windowed_per_axis(device):
    model = cc.WindowedConv2d(C, C, (3, 5), window=(None, "hann"))
    window = model.window[0, 0]
    assert torch.equal(window[0], window[1]), "no window along height"
    assert torch.allclose(window[0], cc.get_window("hann", 5))
    with pytest.raises(ValueError):
        cc.WindowedConv2d(C, C, 3, window=("hann", ))


def test_windowed_gradient_reaches_weight():
    with torch.enable_grad():
        model = cc.WindowedConv1d(C, C, 7, padding="same")
        model(torch.randn(1, C, 64)).sum().backward()
    assert model.weight.grad is not None
    # gradient is scaled by the window: edge taps get less
    grad = model.weight.grad.abs().mean((0, 1))
    assert grad[0] < grad[3] and grad[-1] < grad[3]


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", ["same", "causal"])
def test_windowed_streaming(device, dims, stride, padding):
    kw = conv_kwargs(dims, stride=stride, padding=padding)
    build = lambda: factory("WindowedConv", dims)(C, C, 5, **kw)
    assert cc.test_equal(build, make_input(device, dims))


# ----- modulated convolutions ------------------------------------------------


def reference_modulated(model, x, style, dims):
    """per-sample loop, building each modulated kernel explicitly"""
    weight = model.get_weight()
    out_c, in_g = weight.shape[:2]
    g = model.groups
    scales = model.affine(style)
    outs = []
    for b in range(x.shape[0]):
        # input channel seen by weight[o, i]: group of o, then i
        group = torch.arange(out_c, device=x.device) // (out_c // g)
        channel = group[:, None] * in_g + torch.arange(in_g, device=x.device)
        w = weight * scales[b][channel].view(out_c, in_g, *[1] * dims)
        if model.demodulate:
            norm = w.pow(2).sum(list(range(1, w.dim())), keepdim=True)
            w = w * torch.rsqrt(norm + model.eps)
        reference = factory("Conv", dims)(model.in_channels,
                                          model.out_channels,
                                          model.kernel_size,
                                          **conv_kwargs(dims),
                                          groups=g).to(x.device)
        reference.weight.copy_(w)
        reference.bias.copy_(model.bias)
        outs.append(reference(x[b:b + 1]))
    return torch.cat(outs)


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("demodulate", [True, False])
def test_modulated_matches_reference(device, dims, groups, demodulate):
    model = factory("ModulatedConv", dims)(C, C, 3, style_dim=8,
                                           groups=groups,
                                           demodulate=demodulate,
                                           **conv_kwargs(dims)).to(device)
    x = make_input(device, dims, batch=3)
    style = torch.randn(3, 8, device=device)
    y = model(x, style)
    assert y.shape == x.shape
    assert torch.allclose(y, reference_modulated(model, x, style, dims),
                          1e-4, 1e-4)


def test_modulated_demodulation_normalizes():
    model = cc.ModulatedConv1d(C, 6, 5, style_dim=8, padding="same")
    w = model.modulated_weight(torch.randn(3, 8))
    assert w.shape == (3 * 6, C, 5)
    assert torch.allclose(w.pow(2).sum((1, 2)), torch.ones(18), 1e-4)


def test_modulated_raw_style_and_affine_init():
    # without style_dim, style gives the per-channel scales directly
    raw = cc.ModulatedConv1d(C, C, 3, padding="same", demodulate=False)
    x = torch.randn(2, C, 64)
    plain = cc.Conv1d(C, C, 3, padding="same")
    plain.load_state_dict(raw.state_dict(), strict=False)
    assert torch.allclose(raw(x, torch.ones(2, C)), plain(x), 1e-5, 1e-5)
    # affine bias starts at 1: a zero style keeps the kernel as is
    styled = cc.ModulatedConv1d(C, C, 3, style_dim=8, padding="same",
                                demodulate=False)
    styled.weight.copy_(raw.weight)
    styled.bias.copy_(raw.bias)
    assert torch.allclose(styled(x, torch.zeros(2, 8)), plain(x), 1e-5, 1e-5)


class WithStyle(nn.Module):
    """fixed style, so that cc.test_equal can drive a modulated conv"""

    def __init__(self, conv, style):
        super().__init__()
        self.conv = conv
        self.register_buffer("style", style)
        self.cumulative_delay = conv.cumulative_delay

    def forward(self, x):
        return self.conv(x, self.style)


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", ["same", "causal"])
def test_modulated_streaming(device, dims, stride, padding):
    style = torch.randn(1, 8)
    kw = conv_kwargs(dims, stride=stride, padding=padding)
    build = lambda: WithStyle(
        factory("ModulatedConv", dims)(C, C, 5, style_dim=8, **kw), style)
    assert cc.test_equal(build, make_input(device, dims))


def test_modulated_style_changes_between_chunks(device):
    """each streamed chunk matches the offline output with its own style"""
    n_chunks = 4
    styles = torch.randn(n_chunks, 2, 8, device=device)
    cc.use_cached_conv(False)
    model = cc.ModulatedConv1d(C, C, 9, style_dim=8,
                               padding="same").to(device)
    cc.use_cached_conv(True)
    cmodel = cc.ModulatedConv1d(C, C, 9, style_dim=8,
                                padding="same").to(device)
    cc.use_cached_conv(False)
    cmodel.load_state_dict(model.state_dict())

    x = make_input(device, 1, batch=2)
    chunks = x.chunk(n_chunks, -1)
    streamed = torch.cat([cmodel(c, s) for c, s in zip(chunks, styles)], -1)
    cd = cmodel.cumulative_delay
    size = chunks[0].shape[-1]
    for k in range(n_chunks):
        offline = model(x, styles[k])
        # outputs emitted during chunk k, i.e. offline steps shifted by cd
        start, end = max(k * size - cd, 0), (k + 1) * size - cd
        got = streamed[..., start + cd:end + cd]
        assert torch.allclose(got, offline[..., start:end], 1e-4, 1e-4), k


# ----- upsampling ------------------------------------------------------------


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("factor", [2, 3, 5])
def test_upsample_matches_torch(device, dims, mode, factor):
    model = factory("Upsample", dims)(factor, mode=mode)
    torch_mode = mode if mode == "nearest" else \
        ["linear", "bilinear", "trilinear"][dims - 1]
    reference = nn.Upsample(scale_factor=factor, mode=torch_mode,
                            **({} if mode == "nearest" else
                               {"align_corners": False}))
    x = make_input(device, dims)
    # float64 reference: in float32, F.interpolate drifts along long signals
    # for factors that are not powers of 2 (see cached_conv/upsample.py)
    expected = reference(x.cpu().double()).float().to(device)
    assert torch.allclose(model(x), expected, 1e-5, 1e-5)


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_upsample_streaming(device, dims, mode, factor):
    # spatial axes upsampled too (by 2), time by `factor`
    scale = factor if dims == 1 else (2, ) * (dims - 1) + (factor, )
    build = lambda: factory("Upsample", dims)(scale, mode=mode)
    assert cc.test_equal(build, make_input(device, dims))


@pytest.mark.parametrize("mode,delay", [("nearest", 0), ("linear", 3)])
def test_upsample_delay(device, mode, delay):
    cc.use_cached_conv(True)
    try:
        model = cc.Upsample1d(3, mode=mode)
        assert model.cumulative_delay == delay
        assert cc.Upsample1d(3, mode=mode,
                             cumulative_delay=2).cumulative_delay == 6 + delay
    finally:
        cc.use_cached_conv(False)


def test_upsample_nearest_is_causal(device):
    """nearest needs no delay: chunked output equals offline output"""
    cc.use_cached_conv(True)
    try:
        model = cc.Upsample2d((2, 4))
    finally:
        cc.use_cached_conv(False)
    x = make_input(device, 2)
    expected = nn.Upsample(scale_factor=(2, 4))(x)
    assert torch.equal(cc.chunk_process(model, x, 8), expected)


def test_upsample_in_encoder_decoder(device):
    """strided conv -> linear upsampling -> conv, with delays chained"""

    def build():
        down = cc.Conv1d(C, C, 5, stride=2, padding="same")
        up = cc.Upsample1d(2, mode="linear",
                           cumulative_delay=down.cumulative_delay)
        conv = cc.Conv1d(C, C, 3, padding="same",
                         cumulative_delay=up.cumulative_delay)
        return cc.CachedSequential(down, up, conv)

    x = make_input(device, 1)
    assert cc.test_equal(build, x)
    assert build().to(device)(x).shape == x.shape


def test_upsample_invalid_arguments():
    for cached in (False, True):
        cc.use_cached_conv(cached)
        try:
            with pytest.raises(ValueError):
                cc.Upsample1d(1.5)
            with pytest.raises(ValueError):
                cc.Upsample1d(2, mode="bicubic")
            with pytest.raises(ValueError):
                cc.Upsample2d((2, 2, 2))
        finally:
            cc.use_cached_conv(False)


# ----- TorchScript -----------------------------------------------------------


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_script_variants(device, cached, dims):
    kw = conv_kwargs(dims, stride=2)
    x = make_input(device, dims, batch=2)
    style = torch.randn(2, 8, device=device)

    windowed = factory("WindowedConv", dims)(C, C, 5, **kw).to(device)
    windowed(x)  # caches are allocated on the first call
    scripted = torch.jit.script(windowed)
    assert scripted(x).shape == windowed(x).shape

    modulated = factory("ModulatedConv", dims)(C, C, 5, style_dim=8,
                                               **kw).to(device)
    modulated(x, style)
    scripted = torch.jit.script(modulated)
    assert scripted(x, style).shape == modulated(x, style).shape

    for mode in ("nearest", "linear"):
        upsample = factory("Upsample", dims)(2, mode=mode).to(device)
        upsample(x)
        scripted = torch.jit.script(upsample)
        assert scripted(x).shape == upsample(x).shape
