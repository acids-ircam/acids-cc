# cached_conv

[![Documentation Status](https://readthedocs.org/projects/cached-conv/badge/?version=latest)](https://acids-cc.readthedocs.io/en/latest/)
[![docs](https://github.com/acids-ircam/acids-cc/actions/workflows/docs.yml/badge.svg)](https://github.com/acids-ircam/acids-cc/actions/workflows/docs.yml)

**Cached convolutions for real-time streaming inference with PyTorch.**

`cached_conv` provides drop-in replacements for PyTorch convolutional layers
that process a signal **block by block** and give the same result as
processing it all at once. Each layer keeps the end of the previous block in
an internal cache, used instead of zero padding at block boundaries.

The same model definition can be built in two modes:

- **eager** (default): regular layers, for training on full signals;
- **cached**: streaming layers, for real-time inference, e.g. once exported
  with TorchScript.

## Features

- 1D, 2D and 3D convolutions and transposed convolutions. Time is the last
  axis, the only one that is streamed.
- `"same"`, `"causal"` and `"anticausal"` padding, computed from the kernel
  size and dilation.
- Strides, dilation and groups.
- Latency tracking: every layer exposes its `cumulative_delay`, which
  accumulates across a whole model.
- Windowed convolutions: the kernel is multiplied by a window before
  convolving.
- StyleGAN 2-modulated convolutions. The style can change from one block to
  the next.
- Streamable upsampling (nearest and linear), group normalization and
  alignment of parallel branches.
- TorchScript export, on CPU, CUDA and MPS.

## Installation

```bash
pip install git+https://github.com/acids-ircam/acids-cc.git
```

Requires Python ≥ 3.9 and PyTorch ≥ 1.9.

## Quick start

```python
import torch
import cached_conv as cc


def build_model():
    conv1 = cc.Conv1d(1, 16, 5, padding="same")
    conv2 = cc.Conv1d(16, 1, 5, padding="same",
                      cumulative_delay=conv1.cumulative_delay)
    return cc.CachedSequential(conv1, conv2)


# 1. train in eager mode (the default)
model = build_model()

# 2. build the same model with cached layers, and load the trained weights
cc.use_cached_conv(True)
streaming_model = build_model()
streaming_model.load_state_dict(model.state_dict())

# 3. process a stream block by block
stream = torch.randn(1, 1, 8192)
y = torch.cat([streaming_model(block) for block in stream.split(512, -1)], -1)

# the output is delayed by the model's latency, in samples. The edges differ:
# before the signal starts, eager layers see zero padding, cached layers see
# the previous layer's response to silence
delay = streaming_model.cumulative_delay
offline = model(stream)
assert torch.allclose(y[..., 2 * delay:], offline[..., delay:-delay],
                      atol=1e-5)
```

Cached layers can't look into the future: centered convolutions delay their
output instead. Pass each layer's `cumulative_delay` to the next one so that
the total latency is tracked correctly. With `padding="causal"` everywhere,
the latency only comes from strides.

## Layers

The layers exposed by the package (`cc.Conv1d`, `cc.Upsample2d`...) are
factories: they return a regular or a cached layer, depending on
`cc.use_cached_conv` at construction time.

| | 1D | 2D | 3D |
| --- | --- | --- | --- |
| Convolution | `Conv1d` | `Conv2d` | `Conv3d` |
| Transposed convolution | `ConvTranspose1d` | `ConvTranspose2d` | `ConvTranspose3d` |
| Windowed convolution | `WindowedConv1d` | `WindowedConv2d` | `WindowedConv3d` |
| Modulated convolution | `ModulatedConv1d` | `ModulatedConv2d` | `ModulatedConv3d` |
| Upsampling | `Upsample1d` | `Upsample2d` | `Upsample3d` |
| Branch alignment | `AlignBranches1d` | `AlignBranches2d` | `AlignBranches3d` |
| Group normalization | `CachedGroupNorm1d` | `CachedGroupNorm2d` | `CachedGroupNorm3d` |

Inputs are shaped `(batch, channels, time)`,
`(batch, channels, height, time)` and `(batch, channels, depth, height, time)`.
In 2D and 3D, `padding` sets the mode along time; the other axes are always
padded "same".

```python
# encoder / decoder; the transposed kernel size along time must be
# stride + 2 * padding
down = cc.Conv1d(16, 16, 4, stride=2, padding="same")
up = cc.ConvTranspose1d(16, 16, 4, stride=2, padding=1,
                        cumulative_delay=down.cumulative_delay)

# spectrogram convolution (frequency, time), windowed along time only
conv = cc.WindowedConv2d(1, 16, (3, 33), window=(None, "hann"))

# StyleGAN 2 modulation
mod = cc.ModulatedConv1d(16, 32, 3, padding="same", style_dim=64)
y = mod(torch.randn(2, 16, 512), torch.randn(2, 64))

# upsampling by an integer factor along time
upsample = cc.Upsample1d(2, mode="linear",
                         cumulative_delay=down.cumulative_delay)

# residual connection: delays the identity branch to match the convolution
conv = cc.Conv1d(16, 16, 3, padding="same")
residual = cc.AlignBranches1d(conv, torch.nn.Identity(),
                              delays=[conv.cumulative_delay, 0])
y_conv, y_identity = residual(torch.randn(1, 16, 512))
```

## Exporting with TorchScript

Caches are allocated on the first call. Run the cached model once before
scripting it:

```python
cc.use_cached_conv(True)
model = build_model().eval()

with torch.no_grad():
    model(torch.zeros(1, 1, 512))

torch.jit.script(model).save("model.ts")
```

Caches have room for `max_batch_size` streams (a layer argument). Block
lengths must be divisible by the model's total stride.

## Testing

`cc.test_equal` checks that a model gives the same output when streamed as
when processed offline:

```python
assert cc.test_equal(build_model, torch.randn(1, 1, 4096))
```

To run the library's test suite, on all available devices:

```bash
pip install -e ".[dev]"
pytest tests
```

To run it against several Python / PyTorch versions with
[uv](https://docs.astral.sh/uv/):

```bash
./test_versions.sh -p "3.9 3.12" -t "1.13 2.0 latest"
./test_versions.sh --lowest          # oldest torch for each Python version
```

`tests/test_audio.py` also writes audio examples to `tests/audio_outputs/`
(waveforms in 1D, spectrograms in 2D / 3D), in eager and streaming modes.

## Documentation

The full documentation is on
[cached-conv.readthedocs.io](https://cached-conv.readthedocs.io). To build it
locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Credits

Developed at [ACIDS](https://acids.ircam.fr) (IRCAM) by Axel
Chemla--Romeu-Santos and Nils Demerlé. Upgrade of the original
[cached_conv](https://github.com/acids-ircam/cached_conv) package by Antoine
Caillon (2021).

## License

MIT
