# Installation

From the GitHub repository:

```bash
pip install git+https://github.com/acids-ircam/acids-cc.git
```

or from a local clone, in editable mode, with the test dependencies:

```bash
git clone https://github.com/acids-ircam/acids-cc.git
cd acids-cc
pip install -e ".[dev]"
```

The package imports as `cached_conv`:

```python
import cached_conv as cc
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.9 (tested up to 2.x). Streaming works on CPU, CUDA and MPS.

## Building this documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
