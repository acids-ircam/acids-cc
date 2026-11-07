# Streaming with cached convolutions

## Why caching?

A convolution with kernel size $k$ needs $k - 1$ past (and possibly future)
samples to compute each output. On a full signal these are provided by zero
padding. When the signal arrives block by block, zero-padding each block
creates discontinuities at every block boundary.

Cached layers replace the left padding with the last samples of the
**previous block**, kept in an internal buffer. Processing a stream block by
block then gives exactly the same result as processing the whole signal.

## Eager and cached modes

The layers exposed at the package level (`cc.Conv1d`, `cc.Conv2d`,
`cc.Upsample1d`...) are **factories**: they build a regular (eager) layer or
a cached one, depending on a global flag read *at construction time*.

```python
import cached_conv as cc

cc.use_cached_conv(False)   # default: eager layers, for training
model = build_model()

cc.use_cached_conv(True)    # cached layers, for streaming
streaming_model = build_model()
streaming_model.load_state_dict(model.state_dict())
```

Both versions have the same parameters, so weights trained in eager mode can
be loaded into the cached model. Load them before the first call: caches
are allocated on the first forward pass, and are then part of the cached
model's `state_dict`. `cc.is_cached()` returns the current state of
the flag.

The cached classes can also be used directly (`cc.CachedConv1d`,
`cc.CachedConv2d`, ...), regardless of the flag.

## Tensor layout

Inputs are shaped `(batch, channels, time)` in 1D,
`(batch, channels, height, time)` in 2D and
`(batch, channels, depth, height, time)` in 3D. **Time is always the last
axis**, the only one that is cached. The other axes are processed as usual
and must keep a fixed size from one block to the next.

## Delays and `cumulative_delay`

A cached layer can't look into the future: a centered (`"same"`) convolution
needs $\lfloor k/2 \rfloor$ future samples, so the cached version delays its
output by that amount instead. Strided layers may add a further delay so
that blocks stay aligned on the stride.

Each layer exposes this delay as `cumulative_delay`, counted in output
samples. In a stack of layers, pass the delay of the previous layer to the
next one so that delays accumulate correctly through strides and upsampling:

```python
layers, delay = [], 0
for stride in [1, 2, 2]:
    conv = cc.Conv1d(16, 16, 3, stride=stride, padding="same",
                     cumulative_delay=delay)
    delay = conv.cumulative_delay
    layers.append(conv)

model = cc.CachedSequential(*layers)
model.cumulative_delay  # delay of the whole stack, in output samples
```

{class}`~cached_conv.conv_1d.CachedSequential` takes the
`cumulative_delay` of its last layer that defines one. In eager mode, all
delays are 0.

```{tip}
Use `padding="causal"` everywhere for zero-latency models. With causal
layers, `cumulative_delay` only comes from strides.
```

## Parallel branches

When several branches with different delays are combined (e.g. a residual
connection around a centered convolution), they must be aligned.
`cc.AlignBranches1d` (and `2d` / `3d`) delays each branch so that they all
match the slowest one:

```python
conv = cc.Conv1d(16, 16, 3, padding="same")
residual = cc.AlignBranches1d(conv, torch.nn.Identity(),
                              delays=[conv.cumulative_delay, 0])

y_conv, y_identity = residual(x)
y = y_conv + y_identity
```

In eager mode, `AlignBranches` simply returns the outputs of all branches.

## Block sizes and batch size

- Block lengths must be divisible by the total stride of the model.
- Caches are allocated on the first call, with room for `max_batch_size`
  streams (a layer argument). Any batch size up to that value can be used
  afterwards. Each batch slot keeps its own stream.

## Exporting with TorchScript

Caches are created on the first forward pass. **Call the model once before
scripting it**:

```python
cc.use_cached_conv(True)
model = build_model().eval()

with torch.no_grad():
    model(torch.zeros(1, 1, 2048))     # allocates the caches

scripted = torch.jit.script(model)
scripted.save("model.ts")
```

```{warning}
The first call also writes into the caches. Reset the model state (or build
a fresh model) if this warm-up input must not affect the stream.
```

## Devices

Caches are registered buffers, so `model.to(device)` moves them along with
the parameters, including in the middle of a stream.
