import torch
import torch.nn as nn

import cached_conv as cc

torch.set_grad_enabled(False)


# ----- 1D -----

def test_residual_1d():
    cc.use_cached_conv(False)
    conv = cc.Conv1d(1, 1, 5, padding="same")
    model = cc.AlignBranches1d(conv, nn.Identity(), delays=[conv.cumulative_delay, 0])

    cc.use_cached_conv(True)
    cconv = cc.Conv1d(1, 1, 5, padding="same")
    cmodel = cc.AlignBranches1d(cconv, nn.Identity(), delays=[cconv.cumulative_delay, 0])

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 2**14)
    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_1d():
    cc.use_cached_conv(False)
    b1 = cc.Conv1d(1, 1, 5, padding="same")
    b2 = cc.Conv1d(1, 1, 3, padding="same")
    model = cc.AlignBranches1d(b1, b2)

    cc.use_cached_conv(True)
    cb1 = cc.Conv1d(1, 1, 5, padding="same")
    cb2 = cc.Conv1d(1, 1, 3, padding="same")
    cmodel = cc.AlignBranches1d(cb1, cb2)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 2**14)
    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_stride_1d():
    cc.use_cached_conv(False)
    b1 = cc.Conv1d(1, 1, 5, stride=2, padding="same")
    b2 = cc.Conv1d(1, 1, 3, stride=2, padding="same")
    model = cc.AlignBranches1d(b1, b2, stride=.5)

    cc.use_cached_conv(True)
    cb1 = cc.Conv1d(1, 1, 5, stride=2, padding="same")
    cb2 = cc.Conv1d(1, 1, 3, stride=2, padding="same")
    cmodel = cc.AlignBranches1d(cb1, cb2, stride=.5)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 2**14)
    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_transpose_1d():
    cc.use_cached_conv(False)
    b1 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    b2 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    model = cc.AlignBranches1d(b1, b2, stride=2)

    cc.use_cached_conv(True)
    cb1 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    cb2 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    cmodel = cc.AlignBranches1d(cb1, cb2, stride=2)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 2**14)
    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


# ----- 2D -----

def test_residual_2d():
    cc.use_cached_conv(False)
    conv = cc.Conv2d(1, 1, 3, padding="same")
    model = cc.AlignBranches2d(conv, nn.Identity(), delays=[conv.cumulative_delay, 0])

    cc.use_cached_conv(True)
    cconv = cc.Conv2d(1, 1, 3, padding="same")
    cmodel = cc.AlignBranches2d(cconv, nn.Identity(), delays=[cconv.cumulative_delay, 0])

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 8, 2**14)
    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_2d():
    cc.use_cached_conv(False)
    b1 = cc.Conv2d(1, 1, 5, padding="same")
    b2 = cc.Conv2d(1, 1, 3, padding="same")
    model = cc.AlignBranches2d(b1, b2)

    cc.use_cached_conv(True)
    cb1 = cc.Conv2d(1, 1, 5, padding="same")
    cb2 = cc.Conv2d(1, 1, 3, padding="same")
    cmodel = cc.AlignBranches2d(cb1, cb2)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 8, 2**14)
    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_stride_2d():
    cc.use_cached_conv(False)
    b1 = cc.Conv2d(1, 1, 5, stride=2, padding="same")
    b2 = cc.Conv2d(1, 1, 3, stride=2, padding="same")
    model = cc.AlignBranches2d(b1, b2, stride=.5)

    cc.use_cached_conv(True)
    cb1 = cc.Conv2d(1, 1, 5, stride=2, padding="same")
    cb2 = cc.Conv2d(1, 1, 3, stride=2, padding="same")
    cmodel = cc.AlignBranches2d(cb1, cb2, stride=.5)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 8, 2**14)
    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_transpose_2d():
    cc.use_cached_conv(False)
    b1 = cc.ConvTranspose2d(1, 1, (1, 4), (1, 2), (0, 1))
    b2 = cc.ConvTranspose2d(1, 1, (1, 4), (1, 2), (0, 1))
    model = cc.AlignBranches2d(b1, b2, stride=2)

    cc.use_cached_conv(True)
    cb1 = cc.ConvTranspose2d(1, 1, (1, 4), (1, 2), (0, 1))
    cb2 = cc.ConvTranspose2d(1, 1, (1, 4), (1, 2), (0, 1))
    cmodel = cc.AlignBranches2d(cb1, cb2, stride=2)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 8, 2**14)
    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(lambda x: sum(cmodel(x)), x, 4)[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)
