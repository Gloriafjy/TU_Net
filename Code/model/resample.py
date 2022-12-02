import math
import torch as th
from torch.nn import functional as F


def sinc(t):
    return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_upsample2(zeros=56):
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    *other, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = th.stack([x, out], dim=-1)
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)
