import numpy as np
import torch
from torch.nn import functional as F


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_frequencies(n_mels, fmin, fmax):
    low = hz_to_mel(fmin)
    high = hz_to_mel(fmax)
    mels = np.linspace(low, high, n_mels)
    return mel_to_hz(mels)


class LowPassFilters(torch.nn.Module):
    def __init__(self, cutoffs: list, width: int = None):
        super().__init__()
        self.cutoffs = cutoffs
        if width is None:
            width = int(2 / min(cutoffs))
        self.width = width
        window = torch.hamming_window(2 * width + 1, periodic=False)
        t = np.arange(-width, width + 1, dtype=np.float32)
        filters = []
        for cutoff in cutoffs:
            sinc = torch.from_numpy(np.sinc(2 * cutoff * t))
            filters.append(2 * cutoff * sinc * window)
        self.register_buffer("filters", torch.stack(filters).unsqueeze(1))

    def forward(self, input):
        *others, t = input.shape
        input = input.view(-1, 1, t)
        out = F.conv1d(input, self.filters, padding=self.width)
        return out.permute(1, 0, 2).reshape(-1, *others, t)

    def __repr__(self):
        return "LossPassFilters(width={},cutoffs={})".format(self.width, self.cutoffs)
