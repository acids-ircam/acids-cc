"""
Moving a model across devices *after* its caches were created.

Caches are lazily allocated on the first forward pass, on the device of the
input. A model warmed up on CPU then moved with .to(device) must carry its
caches along (they are registered buffers) and keep streaming seamlessly.
"""
import copy

import pytest
import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)

C = 4


def layers():
    return {
        "conv1d": (lambda: cc.CachedConv1d(C, C, 5, stride=2, padding="same"),
                   (1, C, 512)),
        "conv_t1d": (lambda: cc.CachedConvTranspose1d(C, C, 4, 2, 1),
                     (1, C, 512)),
        "padding": (lambda: cc.CachedPadding(7), (1, C, 512)),
        "align_branches1d": (lambda: cc._AlignBranches1d(
            cc.CachedConv1d(C, C, 5, padding="same"), nn.Identity(),
            delays=[2, 0]), (1, C, 512)),
        "group_norm": (lambda: cc.CachedGroupNorm(2, C, max_batch_size=1),
                       (1, C, 512)),
        "conv2d": (lambda: cc.CachedConv2d(C, C, 3, stride=(1, 2)),
                   (1, C, 8, 512)),
        "conv_t2d": (lambda: cc.CachedConvTranspose2d(
            C, C, (1, 4), (1, 2), (0, 1)), (1, C, 8, 512)),
        "conv3d": (lambda: cc.CachedConv3d(C, C, 3, stride=(1, 1, 2)),
                   (1, C, 3, 5, 512)),
        "conv_t3d": (lambda: cc.CachedConvTranspose3d(
            C, C, (1, 3, 4), (1, 1, 2), (0, 1, 1)), (1, C, 3, 5, 512)),
    }


def run(model, x):
    y = model(x)
    return torch.cat(y, 1) if isinstance(y, (list, tuple)) else y


@pytest.fixture(autouse=True)
def cached_mode():
    state = cc.USE_CACHED_CONV
    cc.use_cached_conv(True)  # CachedConvTranspose2d reads it at build time
    yield
    cc.use_cached_conv(state)


@pytest.mark.parametrize("name", list(layers()))
def test_move_after_warmup(device, name):
    build, shape = layers()[name]
    torch.manual_seed(0)
    reference = build()
    moved = copy.deepcopy(reference)

    chunks = torch.randn(*shape).chunk(4, -1)

    # reference: everything on CPU
    expected = torch.cat([run(reference, c) for c in chunks], -1)

    # warm up on CPU (allocates the caches there), then move mid-stream
    first = run(moved, chunks[0])
    moved.to(device)
    for t in list(moved.parameters()) + list(moved.buffers()):
        assert t.device.type == device.type, "a cache was left behind"
    rest = [run(moved, c.to(device)) for c in chunks[1:]]
    assert all(r.device.type == device.type for r in rest)

    streamed = torch.cat([first] + [r.cpu() for r in rest], -1)
    assert torch.allclose(expected, streamed, 1e-4, 1e-4)
