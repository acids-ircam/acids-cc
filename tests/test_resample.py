"""Tests of cc.Resample1d / CachedResample1d.

Rates cover integer down / up ratios (power of 2 or not) and rational ratios,
in both modes (eager, whole signal / cached, block streaming):

- output lengths, for any input length (eager) or chunk length (cached);
- streaming == eager once ``time_delay`` is compensated, also after
  TorchScript compilation;
- signal quality: unit DC gain, passband tones kept, aliasing (down) and
  imaging (up) rejected.
"""
import math

import pytest
import torch

import cached_conv as cc
from cached_conv.resample import CachedResample1d, Resample1d

torch.set_grad_enabled(False)

# (orig_sr, target_sr)
INTEGER = [(44100, 22050), (88200, 22050), (176400, 22050), (66150, 22050),
           (22050, 44100), (22050, 88200), (22050, 66150), (48000, 16000)]
RATIONAL = [(48000, 44100), (44100, 48000), (16000, 22050), (32000, 48000),
            (48000, 32000)]
RATES = INTEGER + RATIONAL
IDS = [f"{o}->{t}" for o, t in RATES]


@pytest.fixture
def cached_state():
    initial = cc.is_cached()
    yield
    cc.use_cached_conv(initial)


def build(orig_sr, target_sr, cached, **kwargs):
    cc.use_cached_conv(cached)
    return cc.Resample1d(orig_sr, target_sr, **kwargs)


def n_chunks_input(module, n_chunks=6, frames=37):
    """input length made of ``n_chunks`` chunks, multiples of ``down``"""
    return n_chunks, frames * module.down


def tone(freq, sr, length, phase=0.):
    t = torch.arange(length, dtype=torch.float64) / sr
    return torch.sin(2 * math.pi * freq * t + phase).float()


def rms_db(x):
    return 20 * math.log10(x.pow(2).mean().sqrt().item() + 1e-12)


# ----- factory / construction ------------------------------------------------

@pytest.mark.parametrize("cached", [False, True], ids=["eager", "cached"])
def test_factory(cached_state, cached):
    m = build(44100, 22050, cached)
    assert isinstance(m, CachedResample1d if cached else Resample1d)
    assert isinstance(cc.Resample(44100, 22050), type(m))


@pytest.mark.parametrize("orig_sr,target_sr", RATES, ids=IDS)
def test_ratio(orig_sr, target_sr):
    m = Resample1d(orig_sr, target_sr)
    assert m.up * orig_sr == m.down * target_sr
    assert math.gcd(m.up, m.down) == 1
    assert m.right % m.down == 0


def test_invalid_rates():
    with pytest.raises(ValueError):
        Resample1d(44100.5, 22050)
    with pytest.raises(ValueError):
        Resample1d(0, 22050)


@pytest.mark.parametrize("cached", [False, True], ids=["eager", "cached"])
def test_identity(cached_state, cached):
    m = build(44100, 44100, cached)
    x = torch.randn(2, 1, 1024)
    assert torch.equal(m(x), x)
    assert m.cumulative_delay == 0


# ----- shapes ----------------------------------------------------------------

@pytest.mark.parametrize("orig_sr,target_sr", RATES, ids=IDS)
@pytest.mark.parametrize("length", [1, 999, 4096, 4097])
def test_eager_any_length(orig_sr, target_sr, length):
    m = Resample1d(orig_sr, target_sr)
    x = torch.randn(2, 3, length)
    y = m(x)
    assert y.shape == (2, 3, math.ceil(length * m.up / m.down))


@pytest.mark.parametrize("orig_sr,target_sr", RATES, ids=IDS)
def test_cached_chunk_length(orig_sr, target_sr):
    m = CachedResample1d(orig_sr, target_sr)
    x = torch.randn(2, 3, 5 * m.down)
    assert m(x).shape == (2, 3, 5 * m.up)
    if m.down > 1:
        with pytest.raises(ValueError):
            m(torch.randn(2, 3, 5 * m.down + 1))


# ----- streaming == offline --------------------------------------------------

def compare_streaming(eager, cached, x, n_chunks):
    y = eager(x)
    chunks = torch.split(x, x.shape[-1] // n_chunks, -1)
    cy = torch.cat([cached(c) for c in chunks], -1)
    assert cy.shape == y.shape
    d = cached.time_delay
    assert d < y.shape[-1], "signal too short w.r.t. the resampler delay"
    torch.testing.assert_close(cy[..., d:], y[..., :y.shape[-1] - d],
                               rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("orig_sr,target_sr", RATES, ids=IDS)
def test_streaming_equals_eager(device, orig_sr, target_sr):
    eager = Resample1d(orig_sr, target_sr).to(device)
    cached = CachedResample1d(orig_sr, target_sr).to(device)
    n_chunks, chunk = n_chunks_input(cached)
    x = torch.randn(3, 2, n_chunks * chunk, device=device)
    compare_streaming(eager, cached, x, n_chunks)


@pytest.mark.parametrize("orig_sr,target_sr", [(44100, 22050), (22050, 44100),
                                               (48000, 44100)],
                         ids=["down2", "up2", "rational"])
def test_streaming_smaller_batch(orig_sr, target_sr):
    """chunks with a batch smaller than max_batch_size use the first slots"""
    cached = CachedResample1d(orig_sr, target_sr, max_batch_size=4)
    eager = Resample1d(orig_sr, target_sr)
    n_chunks, chunk = n_chunks_input(cached)
    x = torch.randn(1, 1, n_chunks * chunk)
    compare_streaming(eager, cached, x, n_chunks)


@pytest.mark.parametrize("orig_sr,target_sr", [(44100, 22050), (22050, 44100),
                                               (66150, 22050),
                                               (44100, 48000)],
                         ids=["down2", "up2", "down3", "rational"])
def test_script_streaming(orig_sr, target_sr):
    eager = torch.jit.script(Resample1d(orig_sr, target_sr))
    cached = CachedResample1d(orig_sr, target_sr)
    n_chunks, chunk = n_chunks_input(cached)
    # allocates the cache (zeros in, so it stays at its initial state)
    cached(torch.zeros(1, 1, chunk))
    scripted = torch.jit.script(cached)
    assert scripted.time_delay == cached.time_delay
    x = torch.randn(1, 1, n_chunks * chunk)
    compare_streaming(eager, scripted, x, n_chunks)


def test_cumulative_delay_chain():
    """down then up: delays add up, in output samples of each layer"""
    down = CachedResample1d(44100, 22050)
    up = CachedResample1d(22050, 44100,
                          cumulative_delay=down.cumulative_delay)
    assert up.cumulative_delay == 2 * down.time_delay + up.time_delay

    edown, eup = Resample1d(44100, 22050), Resample1d(22050, 44100)
    x = torch.randn(1, 1, 16 * 512)
    y = eup(edown(x))
    cy = torch.cat([up(down(c)) for c in torch.split(x, 512, -1)], -1)
    d = up.cumulative_delay
    # the eager chain drops the pre-ringing of `down` before t=0, that the
    # streaming one feeds to `up`: start comparing after the up filter span
    c = up.kernel.shape[-1] * up.up
    torch.testing.assert_close(cy[..., d + c:], y[..., c:-d], rtol=1e-4,
                               atol=1e-5)


# ----- signal quality --------------------------------------------------------

@pytest.mark.parametrize("orig_sr,target_sr", RATES, ids=IDS)
def test_dc_gain(orig_sr, target_sr):
    m = Resample1d(orig_sr, target_sr)
    torch.testing.assert_close(m.kernel.sum(-1),
                               torch.ones(m.up, 1), rtol=0, atol=1e-5)
    y = m(torch.ones(1, 1, 8 * m.down + 4096))
    # away from the zero-padded edges
    torch.testing.assert_close(y[..., 200:-200],
                               torch.ones_like(y[..., 200:-200]),
                               rtol=0, atol=1e-4)


def steady(y, margin):
    return y[..., margin:-margin]


@pytest.mark.parametrize("orig_sr,target_sr", RATES, ids=IDS)
@pytest.mark.parametrize("cached", [False, True], ids=["eager", "cached"])
def test_passband(cached_state, orig_sr, target_sr, cached):
    """a tone at half of the lowest Nyquist goes through at 0 dB, at the
    right frequency"""
    m = build(orig_sr, target_sr, cached)
    nyq = min(orig_sr, target_sr) / 2
    freq = .5 * nyq
    length = m.down * math.ceil(.25 * orig_sr / m.down)
    x = tone(freq, orig_sr, length)[None, None]
    y = m(x)
    d = getattr(m, "time_delay", 0)
    ref = tone(freq, target_sr, y.shape[-1] - d)[None, None]
    margin = 2 * m.kernel.shape[-1]
    assert abs(rms_db(steady(y[..., d:], margin)) -
               rms_db(steady(ref, margin))) < .05
    err = steady(y[..., d:], margin) - steady(ref, margin)
    assert rms_db(err) < -60


DOWN = [(o, t) for o, t in RATES if t < o]
UP = [(o, t) for o, t in RATES if t > o]


@pytest.mark.parametrize("orig_sr,target_sr", DOWN,
                         ids=[f"{o}->{t}" for o, t in DOWN])
@pytest.mark.parametrize("cached", [False, True], ids=["eager", "cached"])
def test_antialiasing(cached_state, orig_sr, target_sr, cached):
    """tones above the target Nyquist are removed instead of folding back"""
    m = build(orig_sr, target_sr, cached)
    margin = 2 * m.kernel.shape[-1]
    length = m.down * math.ceil(.25 * orig_sr / m.down)
    nyq_out, nyq_in = target_sr / 2, orig_sr / 2
    # between the target Nyquist (+ 2% transition) and the input one
    for a in [0., .3, .9]:
        freq = 1.02 * nyq_out + a * (nyq_in - 1.02 * nyq_out)
        x = tone(freq, orig_sr, length)[None, None]
        y = m(x)
        assert rms_db(steady(y, margin)) < -80, f"{freq} Hz aliased"


@pytest.mark.parametrize("orig_sr,target_sr", UP,
                         ids=[f"{o}->{t}" for o, t in UP])
@pytest.mark.parametrize("cached", [False, True], ids=["eager", "cached"])
def test_anti_imaging(cached_state, orig_sr, target_sr, cached):
    """upsampling creates no energy above the original Nyquist"""
    m = build(orig_sr, target_sr, cached)
    margin = 2 * m.kernel.shape[-1]
    length = m.down * math.ceil(.5 * orig_sr / m.down)
    nyq_in = orig_sr / 2
    for freq in [.1 * nyq_in, .5 * nyq_in, .8 * nyq_in]:
        x = tone(freq, orig_sr, length)[None, None]
        y = steady(m(x), margin)[0, 0].double()
        y = y * torch.hann_window(y.shape[-1], dtype=torch.float64)
        spec = torch.fft.rfft(y).abs()
        freqs = torch.fft.rfftfreq(y.shape[-1], 1 / target_sr)
        image = spec[freqs > nyq_in * 1.02].max()
        assert 20 * torch.log10(image / spec.max()) < -80, \
            f"{freq} Hz imaged"


def test_roundtrip():
    """down then up gives back a band-limited signal"""
    down, up = Resample1d(44100, 22050), Resample1d(22050, 44100)
    x = sum(tone(f, 44100, 2**14, p) for f, p in [(220, 0), (1234, 1.),
                                                     (7000, 2.)])[None, None]
    y = up(down(x))
    assert y.shape == x.shape
    m = 1024
    assert rms_db(steady(y - x, m)) - rms_db(steady(x, m)) < -60
