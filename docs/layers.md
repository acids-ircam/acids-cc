# Layers

Every layer below is available as a factory at the package level
(`cc.<Name>`). The factory returns the eager or the cached class depending on
{func}`cached_conv.use_cached_conv` (see {doc}`streaming`).

| Factory | 1D | 2D | 3D |
|---|---|---|---|
| Convolution | `Conv1d` | `Conv2d` | `Conv3d` |
| Transposed convolution | `ConvTranspose1d` | `ConvTranspose2d` | `ConvTranspose3d` |
| Windowed convolution | `WindowedConv1d` | `WindowedConv2d` | `WindowedConv3d` |
| Modulated convolution | `ModulatedConv1d` | `ModulatedConv2d` | `ModulatedConv3d` |
| Upsampling | `Upsample1d` | `Upsample2d` | `Upsample3d` |
| Branch alignment | `AlignBranches1d` | `AlignBranches2d` | `AlignBranches3d` |

All of them accept the extra keyword arguments `cumulative_delay` and
`max_batch_size`, which eager layers ignore.

## Convolutions

`Conv1d`, `Conv2d` and `Conv3d` take the arguments of their
{mod}`torch.nn` counterparts. `padding` computes the padding from the kernel
size and dilation:

| `padding` | padding along time | delay in cached mode (stride 1) |
|---|---|---|
| `"same"` / `"centered"` | centered | ⌈(k − 1)·d / 2⌉ samples |
| `"causal"` | left only | none |
| `"anticausal"` | right only | (k − 1)·d samples |

($k$: kernel size, $d$: dilation. With a stride, the delay is rounded up to a multiple of the stride and
counted in output samples.)

- **1D**: `padding` also accepts `"valid"`, an int, or a `(left, right)`
  pair. The default is `0`, as in PyTorch.
- **2D / 3D**: the default is `"same"`. `padding` only sets the mode along
  time; the other axes are always padded "same" (centered), so their sizes
  are kept with stride 1.

```python
cc.Conv1d(1, 16, 7, padding="causal")
cc.Conv2d(1, 16, (3, 5), stride=(1, 2))                # (freq, time)
cc.Conv3d(2, 16, (3, 3, 5), dilation=(1, 1, 4))        # (depth, freq, time)
```

## Transposed convolutions

`ConvTranspose1d/2d/3d` use the regular PyTorch arguments. In cached mode,
blocks are joined by overlap-add. For the output to be exactly `stride`
times longer than the input, the kernel size along time must satisfy

$$\text{kernel\_size} = \text{stride} + 2 \cdot \text{padding}$$

```python
down = cc.Conv1d(16, 16, 4, stride=2, padding="same")
up = cc.ConvTranspose1d(16, 16, 4, stride=2, padding=1,
                        cumulative_delay=down.cumulative_delay)
```

For 2D / 3D, `padding` is a regular `(…, time)` tuple; only its time value
is cached.

## Group normalization

{class}`~cached_conv.conv_1d.CachedGroupNorm` (aliases `CachedGroupNorm1d`,
`CachedGroupNorm2d`, `CachedGroupNorm3d`) computes its statistics on the
current block plus a history of `history_size` previous samples. By default,
the history has the length of the first block. Its output depends on the
block size, so it does not match `nn.GroupNorm` over the full signal. Call
`reset_cache()` before starting a new stream.

## Windowed convolutions

`WindowedConv{1,2,3}d` multiply their kernel by a fixed window before
convolving, which gives smoother, band-limited filters. They take an extra
`window` argument:

- a name: `"hann"` (default), `"hamming"`, `"blackman"`, `"bartlett"`,
  `"kaiser"`. Each is computed on `kernel_size + 2` points and its two end
  points are dropped, so no tap is zeroed: a size-3 Hann kernel is scaled by
  `[.5, 1, .5]`;
- `None` / `"rect"`: no windowing;
- a callable `f(size) -> Tensor`, or explicit values;
- a **tuple** with one window per kernel axis. For instance, to window a 2D
  kernel along time only:

```python
cc.WindowedConv2d(1, 16, (3, 33), window=(None, "hann"))
```

See {func}`cached_conv.variants.get_window`.

## Modulated convolutions

`ModulatedConv{1,2,3}d` implement StyleGAN 2 weight modulation and
demodulation. The kernel is scaled per sample and input channel by a style
vector, then normalized per output channel:

```python
conv = cc.ModulatedConv1d(16, 32, 3, padding="same", style_dim=64)
y = conv(x, style)          # x: (batch, 16, time), style: (batch, 64)
```

- With `style_dim`, a learned linear layer maps the style to one scale per
  input channel. Its bias is initialized to 1. Without it, `style` directly
  holds the scales, shaped `(batch, in_channels)`.
- `demodulate=True` (default) normalizes each output channel's kernel to unit
  norm.
- The whole batch is computed as a single grouped convolution, and `groups`
  is supported.
- In streaming mode, the style can change from one block to the next.

A modulated layer takes two inputs, so put it inside your own module rather
than in a `CachedSequential`.

## Upsampling

`Upsample{1,2,3}d(scale_factor, mode="nearest")` upsample a signal and can
be streamed. `scale_factor` is a number or one value per axis
`(…, time)`. **The time factor must be an integer.**

- `mode="nearest"`: samples are repeated. Causal, with no delay.
- `mode="linear"` (also `"bilinear"` / `"trilinear"`): linear interpolation
  with `align_corners=False`, as with {class}`torch.nn.Upsample`. Each output
  depends on the next input sample, so the cached version adds one input
  sample of delay (`scale_factor` output samples).

```python
down = cc.Conv1d(16, 16, 4, stride=2, padding="causal")
up = cc.Upsample1d(2, mode="linear", cumulative_delay=down.cumulative_delay)
```

Along time, linear interpolation is computed with exact per-phase weights.
The eager and cached versions share this code, which avoids a float32
precision drift of `F.interpolate` on long signals.
