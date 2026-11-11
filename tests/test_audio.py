"""
Listenable end-to-end tests.

A synthetic test signal (chirp + harmonic notes + noise bursts) goes through a
model designed to *give back its input*, so that any defect is audible. Each
test changes one convolution hyperparameter from the default
(padding="same", kernel_size=9, stride=1, dilation=1, groups=1).

The model is run once offline (USE_CACHED_CONV=False, whole signal at once)
and once in streaming mode (USE_CACHED_CONV=True, fed block by block like an
audio callback). We check that

- both modes match once the streaming latency is compensated,
- the output resynthesizes the input (band-limited version if stride > 1),
- a TorchScript version of the cached model streams the same.

Model, for each config:

- kernel = delta on the tap aligning output and input, plus a small spread
  over every other tap: the output is nearly the input, yet every tap (and
  so every cached sample) matters. Cache errors would sound like clicks at
  block boundaries.
- the mono signal is spread over C channels with different gains, each
  channel being mapped to itself inside its group: mixing channels across
  groups breaks the resynthesis.
- stride > 1: anti-aliasing strided conv, then interpolating transposed
  conv. The output is the input band-limited to sr / (2 * stride).

1d tests process the raw waveform, 2d tests the STFT magnitude, that is
resynthesized with the original phase. 3d tests process a stereo STFT
magnitude (depth = audio channel), the right channel being a delayed, quieter
copy of the left one: mixing or misaligning the depth axis would be heard as
a wrong stereo image. Files are written to
tests/audio_outputs/<device>/ (override root with $CC_AUDIO_DIR). Run with
`pytest tests/test_audio.py -s` to print the paths.
"""
import math
from array import array
import os
import wave
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)

SR = 16000
N_SAMPLES = 3 * 2**14  # ~3s, divisible by every block size below
BLOCK_1D = 512  # samples per streaming block
N_FFT, HOP = 512, 128
BLOCK_2D = 8  # STFT frames per streaming block
C = 4  # channels the mono signal is spread over (for groups)
DEC_TAPS = 17  # interpolation taps per output phase (stride > 1)
DEC_PAD = DEC_TAPS // 2  # so that kernel = stride + 2 * padding
OUT_DIR = Path(os.environ.get("CC_AUDIO_DIR",
                              Path(__file__).parent / "audio_outputs"))

DEFAULT = dict(padding="same", kernel_size=9, stride=1, dilation=1, groups=1)
VARIATIONS = {
    "padding": ["same", "causal", "anticausal"],
    "kernel_size": [1, 3, 33],
    "stride": [2, 4],
    "dilation": [2, 8],
    "groups": [2, 4],
}
# strided models need a longer kernel for a decent anti-aliasing filter
CONFIGS = {
    f"{axis}={value}": {
        **DEFAULT,
        **({"kernel_size": 65} if axis == "stride" else {}),
        axis: value,
    }
    for axis, values in VARIATIONS.items() for value in values
}

# ----- utils -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_cache_flag():
    state = cc.USE_CACHED_CONV
    yield
    cc.use_cached_conv(state)


def make_signal():
    g = torch.Generator().manual_seed(0)
    t = torch.arange(N_SAMPLES) / SR
    duration = N_SAMPLES / SR

    # exponential chirp 100Hz -> 7kHz
    f0, f1 = 100., 7000.
    k = math.log(f1 / f0) / duration
    chirp = .15 * torch.sin(2 * math.pi * f0 * (torch.exp(k * t) - 1) / k)

    # harmonic notes (A minor arpeggio), 12 harmonics each, decaying envelope
    notes = torch.zeros_like(t)
    note_len = N_SAMPLES // 12
    for i, midi in enumerate([57, 60, 64, 69] * 3):
        f = 440 * 2**((midi - 69) / 12)
        tt = t[:note_len]
        env = torch.exp(-tt * 4)
        tone = sum(torch.sin(2 * math.pi * f * h * tt) / h
                   for h in range(1, 13) if f * h < SR / 2)
        notes[i * note_len:(i + 1) * note_len] = .2 * env * tone

    # short white noise bursts every 0.5s (transients / high frequencies)
    noise = torch.zeros_like(t)
    burst = SR // 50
    for start in range(SR // 4, N_SAMPLES - burst, SR // 2):
        noise[start:start + burst] = .3 * torch.randn(burst, generator=g) \
            * torch.hann_window(burst)

    return (chirp + notes + noise)[None, None]  # (1, 1, N)


def write_wav(name, x):
    """x: (..., n_channels, n_samples) or (n_samples, )"""
    n_channels = x.shape[-2] if x.dim() > 1 else 1
    x = x.cpu().reshape(n_channels, -1).t().reshape(-1).clamp(-1, 1)
    pcm = (x * 32767).round().to(torch.int16)
    path = OUT_DIR / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(n_channels)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(array("h", pcm.tolist()).tobytes())  # no numpy needed
    print(f"  wrote {path}")
    return path


def stream(model, x, block):
    """Feeds x to model block by block, as a real-time audio callback would."""
    return torch.cat([model(chunk) for chunk in x.split(block, -1)], -1)


def stream_scripted(model, x, block):
    """Same as stream, through a TorchScript version of the model.

    Caches are allocated on the first call, which cannot be scripted
    (torch.jit.unused): the first block warms the eager model up, the
    following ones go through the scripted model.
    """
    chunks = x.split(block, -1)
    first = model(chunks[0])
    scripted = torch.jit.script(model)
    return torch.cat([first] + [scripted(c) for c in chunks[1:]], -1)


def compensate(y, delay):
    """Removes streaming latency, zero-padding the end to keep the length."""
    if delay == 0:
        return y
    return nn.functional.pad(y[..., delay:], (0, delay))


def assert_streaming_matches(offline, streamed, delay, atol=1e-4):
    aligned = compensate(streamed, delay)
    # - the last `delay` samples cannot be produced yet in streaming mode
    # - the first ones may differ: in streaming, non-causal layers emit a
    #   response *before* t=0, that following transposed convs overlap-add
    #   into the beginning (same convention as cached_conv.test_equal)
    end = offline.shape[-1] - delay
    err = (offline[..., delay:end] - aligned[..., delay:end]).abs().max()
    assert err < atol, f"streaming / offline mismatch: max abs error {err}"


def snr_db(y, reference):
    noise = (y - reference).pow(2).sum()
    return 10 * torch.log10(reference.pow(2).sum() / noise).item()


def band_energy_db(x, f_lo, f_hi):
    spec = torch.fft.rfft(x.cpu().reshape(-1)).abs()**2
    freqs = torch.fft.rfftfreq(x.shape[-1], 1 / SR)
    mask = (freqs >= f_lo) & (freqs < f_hi)
    return 10 * torch.log10(spec[mask].sum() + 1e-12).item()


def to_channels(x):
    """mono (1, 1, ...) -> (1, C, ...), channel c scaled by c + 1."""
    extra = (1, ) * (x.dim() - 2)
    scales = torch.arange(1., C + 1, device=x.device).view(1, C, *extra)
    return x.repeat(1, C, *extra) * scales


def from_channels(y):
    """inverse of to_channels: undo the gains, average the channels."""
    extra = (1, ) * (y.dim() - 2)
    scales = torch.arange(1., C + 1, device=y.device).view(1, C, *extra)
    return (y / scales).mean(1, keepdim=True)


def config_id(cfg):
    return "-".join(f"{k}={v}" for k, v in cfg.items())


# ----- kernels ---------------------------------------------------------------


def windowed_sinc(n_taps, cutoff, gain=1.):
    """FIR lowpass. cutoff in cycles / sample (0 < cutoff < .5)."""
    n = torch.arange(n_taps) - (n_taps - 1) / 2
    h = 2 * cutoff * torch.sinc(2 * cutoff * n)
    h = h * torch.blackman_window(n_taps, periodic=False)
    return gain * h / h.sum()


def aligned_tap(kernel_size, dilation, padding):
    """Tap whose input sample is aligned with the output sample."""
    mode = "centered" if padding == "same" else padding
    left, _ = cc.get_padding(kernel_size, dilation=dilation, mode=mode)
    assert left % dilation == 0, "use odd kernel sizes"
    return left // dilation


def near_identity(kernel_size, tap, spread=.1):
    """delta on `tap` + `spread` of the energy smeared over every tap."""
    h = torch.zeros(kernel_size)
    h[tap] = 1
    if kernel_size > 1:
        window = torch.hann_window(kernel_size + 2, periodic=False)[1:-1]
        h += spread * window / window.sum()
    return h


def encoder_kernel(kernel_size, stride, dilation, padding):
    if stride == 1:
        return near_identity(kernel_size,
                             aligned_tap(kernel_size, dilation, padding))
    return windowed_sinc(kernel_size, .45 / stride)  # anti-aliasing


def decoder_kernel(stride):
    """interpolation kernel; size = stride + 2 * padding (cached conv_t)."""
    return windowed_sinc(DEC_TAPS * stride, .5 / stride, gain=stride)


def identity_weight(kernel, groups):
    """(C, C // groups, *kernel) weight mapping each channel to itself."""
    w = torch.zeros(C, C // groups, *kernel.shape)
    for c in range(C):
        w[c, c % (C // groups)] = kernel
    return w


# ----- 1D --------------------------------------------------------------------


def build_1d(padding, kernel_size, stride, dilation, groups):
    enc = cc.Conv1d(C, C, kernel_size, stride=stride, dilation=dilation,
                    groups=groups, padding=padding, bias=False)
    enc.weight.copy_(identity_weight(
        encoder_kernel(kernel_size, stride, dilation, padding), groups))
    if stride == 1:
        return cc.CachedSequential(enc)

    dec = cc.ConvTranspose1d(C, C, DEC_TAPS * stride, stride=stride,
                             padding=DEC_PAD * stride, groups=groups, bias=False,
                             cumulative_delay=enc.cumulative_delay)
    dec.weight.copy_(identity_weight(decoder_kernel(stride), groups))
    return cc.CachedSequential(enc, dec)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_resynthesis_1d(device, name):
    cfg = CONFIGS[name]
    x = make_signal().to(device)
    print(f"\n[1d / {name}] {config_id(cfg)}")
    write_wav(f"{device.type}/1d_input", x)

    cc.use_cached_conv(False)
    offline = from_channels(build_1d(**cfg).to(device)(to_channels(x)))

    cc.use_cached_conv(True)
    model = build_1d(**cfg).to(device)
    delay = model.cumulative_delay
    streamed = from_channels(stream(model, to_channels(x), BLOCK_1D))
    scripted = from_channels(stream_scripted(
        build_1d(**cfg).to(device), to_channels(x), BLOCK_1D))

    print(f"  streaming latency: {delay} samples "
          f"({1000 * delay / SR:.1f} ms) + block of {BLOCK_1D}")
    write_wav(f"{device.type}/1d_{name}_offline", offline)
    write_wav(f"{device.type}/1d_{name}_streaming",
              compensate(streamed, delay))

    assert offline.shape == x.shape == streamed.shape
    assert_streaming_matches(offline, streamed, delay)
    assert_streaming_matches(offline, scripted, delay)
    write_wav(f"{device.type}/1d_{name}_scripted",
              compensate(scripted, delay))

    if cfg["stride"] == 1:
        snr = snr_db(offline, x)
        assert snr > 15, f"input not resynthesized: SNR {snr:.1f} dB"
    else:
        # band-limited resynthesis: low band kept, high band removed
        nyquist = SR / 2 / cfg["stride"]
        low_in, low_out = (band_energy_db(s, 50, .7 * nyquist)
                           for s in (x, offline))
        high_in, high_out = (band_energy_db(s, 1.2 * nyquist, SR / 2)
                             for s in (x, offline))
        assert abs(low_out - low_in) < 1, (low_in, low_out)
        assert high_out < high_in - 30, (high_in, high_out)


# ----- 2D (spectrogram) ------------------------------------------------------


def stft(x):
    window = torch.hann_window(N_FFT, device=x.device)
    spec = torch.stft(x.reshape(1, -1), N_FFT, HOP, window=window,
                      return_complex=True)
    n_frames = spec.shape[-1] // BLOCK_2D * BLOCK_2D
    # contiguous: torch 2.15 + MPS (macOS 14) aborts on repeat() of strided views
    spec = spec[..., :n_frames].contiguous()
    return spec.abs()[:, None], spec.angle()[:, None]  # (1, 1, freq, time)


def istft(mag, phase):
    spec = torch.polar(mag.clamp_min(0)[:, 0], phase[:, 0])
    window = torch.hann_window(N_FFT, device=spec.device)
    return torch.istft(spec, N_FFT, HOP, window=window,
                       length=N_SAMPLES)[None]


def build_2d(padding, kernel_size, stride, dilation, groups):
    # frequency is always centered; time follows `padding`
    enc = cc.Conv2d(C, C, kernel_size, stride=(1, stride),
                    dilation=(dilation, dilation), groups=groups,
                    padding=padding, bias=False)
    # frequency is not cached: pure delta, the spread is only along time
    freq = near_identity(kernel_size,
                         aligned_tap(kernel_size, dilation, "same"), spread=0)
    time = encoder_kernel(kernel_size, stride, dilation, padding)
    enc.weight.copy_(identity_weight(torch.outer(freq, time), groups))
    if stride == 1:
        return cc.CachedSequential(enc)

    dec = cc.ConvTranspose2d(C, C, (1, DEC_TAPS * stride), stride=(1, stride),
                             padding=(0, DEC_PAD * stride), groups=groups,
                             bias=False,
                             cumulative_delay=enc.cumulative_delay)
    dec.weight.copy_(identity_weight(decoder_kernel(stride)[None], groups))
    return cc.CachedSequential(enc, dec)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_resynthesis_2d(device, name):
    cfg = CONFIGS[name]
    x = make_signal().to(device)
    mag, phase = stft(x)
    print(f"\n[2d / {name}] {config_id(cfg)}, spectrogram {tuple(mag.shape)}")
    write_wav(f"{device.type}/2d_input_resynth", istft(mag, phase))

    cc.use_cached_conv(False)
    offline_model = build_2d(**cfg).to(device)
    # ConvTranspose2d crops on the right (crop_right=True): the offline
    # model is itself delayed, like the streaming one
    offline_delay = offline_model.cumulative_delay
    offline = compensate(
        from_channels(offline_model(to_channels(mag))), offline_delay)

    cc.use_cached_conv(True)
    model = build_2d(**cfg).to(device)
    delay = model.cumulative_delay
    streamed = from_channels(stream(model, to_channels(mag), BLOCK_2D))
    scripted = from_channels(stream_scripted(
        build_2d(**cfg).to(device), to_channels(mag), BLOCK_2D))

    print(f"  streaming latency: {delay} frames "
          f"({1000 * delay * HOP / SR:.1f} ms) + block of {BLOCK_2D} frames")
    write_wav(f"{device.type}/2d_{name}_offline", istft(offline, phase))
    write_wav(f"{device.type}/2d_{name}_streaming",
              istft(compensate(streamed, delay), phase))

    assert offline.shape == mag.shape == streamed.shape
    assert_streaming_matches(offline, streamed, delay, atol=1e-3)
    assert_streaming_matches(offline, scripted, delay, atol=1e-3)
    write_wav(f"{device.type}/2d_{name}_scripted",
              istft(compensate(scripted, delay), phase))

    snr = snr_db(offline, mag)
    # strided models smooth the magnitudes along time
    min_snr = 15 if cfg["stride"] == 1 else 5
    assert snr > min_snr, f"input not resynthesized: SNR {snr:.1f} dB"


# ----- 3D (stereo spectrogram) -----------------------------------------------


def make_stereo_signal():
    """left = test signal, right = same signal 250ms later, -2dB."""
    x = make_signal()
    right = .8 * nn.functional.pad(x, (SR // 4, 0))[..., :N_SAMPLES]
    return torch.cat([x, right], 1)  # (1, 2, N)


def stft_stereo(x):
    """(1, 2, N) -> magnitude & phase shaped (1, 1, 2, freq, time)"""
    mags, phases = zip(*(stft(x[:, i:i + 1]) for i in range(x.shape[1])))
    return torch.stack(mags, 2), torch.stack(phases, 2)


def istft_stereo(mag, phase):
    """(1, 1, 2, freq, time) -> (1, 2, N)"""
    return torch.cat([istft(mag[:, :, i], phase[:, :, i])
                      for i in range(mag.shape[2])], 1)


def build_3d(padding, kernel_size, stride, dilation, groups):
    # depth (audio channel) and frequency are always centered; time follows
    # `padding`. Pure delta on depth and frequency, so that the audio
    # channels are not mixed; the spread is only along time (cached axis).
    # kernel_size / dilation vary along time (the streamed axis) only: a
    # full 33x33x33 kernel on a whole spectrogram does not fit in memory
    side = min(kernel_size, 3)
    enc = cc.Conv3d(C, C, (side, side, kernel_size), stride=(1, 1, stride),
                    dilation=(1, 1, dilation), groups=groups,
                    padding=padding, bias=False)
    centered = near_identity(side, aligned_tap(side, 1, "same"), spread=0)
    time = encoder_kernel(kernel_size, stride, dilation, padding)
    kernel = centered[:, None, None] * centered[None, :, None] \
        * time[None, None, :]
    enc.weight.copy_(identity_weight(kernel, groups))
    if stride == 1:
        return cc.CachedSequential(enc)

    dec = cc.ConvTranspose3d(C, C, (1, 1, DEC_TAPS * stride),
                             stride=(1, 1, stride),
                             padding=(0, 0, DEC_PAD * stride), groups=groups,
                             bias=False,
                             cumulative_delay=enc.cumulative_delay)
    dec.weight.copy_(identity_weight(decoder_kernel(stride)[None, None],
                                     groups))
    return cc.CachedSequential(enc, dec)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_resynthesis_3d(device, name):
    cfg = CONFIGS[name]
    x = make_stereo_signal().to(device)
    mag, phase = stft_stereo(x)
    print(f"\n[3d / {name}] {config_id(cfg)}, stereo spectrogram "
          f"{tuple(mag.shape)}")
    write_wav(f"{device.type}/3d_input_resynth", istft_stereo(mag, phase))

    cc.use_cached_conv(False)
    offline = from_channels(build_3d(**cfg).to(device)(to_channels(mag)))

    cc.use_cached_conv(True)
    model = build_3d(**cfg).to(device)
    delay = model.cumulative_delay
    streamed = from_channels(stream(model, to_channels(mag), BLOCK_2D))
    scripted = from_channels(stream_scripted(
        build_3d(**cfg).to(device), to_channels(mag), BLOCK_2D))

    print(f"  streaming latency: {delay} frames "
          f"({1000 * delay * HOP / SR:.1f} ms) + block of {BLOCK_2D} frames")
    write_wav(f"{device.type}/3d_{name}_offline",
              istft_stereo(offline, phase))
    write_wav(f"{device.type}/3d_{name}_streaming",
              istft_stereo(compensate(streamed, delay), phase))

    assert offline.shape == mag.shape == streamed.shape
    assert_streaming_matches(offline, streamed, delay, atol=1e-3)
    assert_streaming_matches(offline, scripted, delay, atol=1e-3)
    write_wav(f"{device.type}/3d_{name}_scripted",
              istft_stereo(compensate(scripted, delay), phase))

    # each audio channel is resynthesized on its own (no stereo mixing)
    min_snr = 15 if cfg["stride"] == 1 else 5
    for i, side in enumerate(["left", "right"]):
        snr = snr_db(offline[:, :, i], mag[:, :, i])
        assert snr > min_snr, f"{side} not resynthesized: SNR {snr:.1f} dB"

