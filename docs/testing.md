# Checking a streaming model

Two helpers compare a model's eager and cached versions.

{func}`cached_conv.chunk_process` runs a model on `N` consecutive blocks of a
signal and concatenates the outputs:

```python
y_stream = cc.chunk_process(cached_model, x, 8)
```

{func}`cached_conv.test_equal` builds a model in eager mode and in cached
mode with the same weights. It then checks that streaming matches offline
processing, once the cached output is shifted by its `cumulative_delay`:

```python
def build():
    first = cc.Conv1d(1, 8, 5, padding="same")
    second = cc.Conv1d(8, 1, 5, stride=2, padding="same",
                       cumulative_delay=first.cumulative_delay)
    return cc.CachedSequential(first, second)

assert cc.test_equal(build, torch.randn(1, 1, 2048))
```

The first and last `cumulative_delay` samples are ignored. They depend on
what comes before the signal starts and after it ends, which is different in
the two modes.

## Library test suite

The repository's own test suite runs on every available device (CPU, CUDA,
MPS):

```bash
pytest tests
```

`test_versions.sh` runs it against several Python and PyTorch versions with
[uv](https://docs.astral.sh/uv/):

```bash
./test_versions.sh -p "3.9 3.12" -t "1.13 2.0 latest"
./test_versions.sh --lowest          # oldest torch for each Python version
```

`tests/test_audio.py` also writes audio examples to `tests/audio_outputs/`.
