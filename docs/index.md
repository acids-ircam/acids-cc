# cached_conv

**Cached convolutions for real-time streaming inference with PyTorch.**

`cached_conv` provides drop-in replacements for PyTorch convolutional layers
that can process a signal **block by block** while giving the same result as
processing it all at once. Each layer keeps the end of the previous block in
an internal cache, which replaces the zero padding a regular convolution
would use at block boundaries.

The same model definition runs in two modes:

- **eager** (default): regular layers, for training on full signals;
- **cached**: streaming layers, for real-time inference, e.g. once exported
  with TorchScript.

```python
import torch
import cached_conv as cc

cc.use_cached_conv(True)            # build streaming layers
conv = cc.Conv1d(1, 16, 5, padding="causal")

stream = torch.randn(1, 1, 4096)
blocks = [conv(block) for block in stream.split(512, -1)]
```

1D (audio), 2D (e.g. spectrograms) and 3D layers are available. The last axis
is always time, the only one that is streamed.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
streaming
layers
testing
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

This package is an upgrade of the original
[cached_conv](https://github.com/acids-ircam/cached_conv) package
(A. Caillon, 2021).
