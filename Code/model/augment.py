import random
import torch as th
from torch import nn
from torch.nn import functional as F
from model import dsp


class Remix(nn.Module):
    def forward(self, sources):
        noise, clean = sources
        bs, *other = noise.shape
        device = noise.device
        perm = th.argsort(th.rand(bs, device=device), dim=0)
        return th.stack([noise[perm], clean])


class RevEcho(nn.Module):
    def __init__(self, proba=0.5, initial=0.3, rt60=(0.3, 1.3), first_delay=(0.01, 0.03),
                 repeat=3, jitter=0.1, keep_clean=0.1, sample_rate=16000):
        super().__init__()
        self.proba = proba
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate

    def _reverb(self, source, initial, first_delay, rt60):
        length = source.shape[-1]
        reverb = th.zeros_like(source)
        for _ in range(self.repeat):
            frac = 1
            echo = initial * source
            while frac > 1e-3:
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                delay = min(
                    1 + int(jitter * first_delay * self.sample_rate),
                    length)
                echo = F.pad(echo[:, :, :-delay], (delay, 0))
                reverb += echo
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                attenuation = 10**(-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation
        return reverb

    def forward(self, wav):
        if random.random() >= self.proba:
            return wav
        noise, clean = wav
        initial = random.random() * self.initial
        first_delay = random.uniform(*self.first_delay)
        rt60 = random.uniform(*self.rt60)

        reverb_noise = self._reverb(noise, initial, first_delay, rt60)
        noise += reverb_noise
        reverb_clean = self._reverb(clean, initial, first_delay, rt60)
        clean += self.keep_clean * reverb_clean
        noise += (1 - self.keep_clean) * reverb_clean

        return th.stack([noise, clean])


class BandMask(nn.Module):
    def __init__(self, maxwidth=0.2, bands=120, sample_rate=16_000):
        super().__init__()
        self.maxwidth = maxwidth
        self.bands = bands
        self.sample_rate = sample_rate

    def forward(self, wav):
        bands = self.bands
        bandwidth = int(abs(self.maxwidth) * bands)
        mels = dsp.mel_frequencies(bands, 40, self.sample_rate/2) / self.sample_rate
        low = random.randrange(bands)
        high = random.randrange(low, min(bands, low + bandwidth))
        filters = dsp.LowPassFilters([mels[low], mels[high]]).to(wav.device)
        low, midlow = filters(wav)
        out = wav - midlow + low
        return out


class Shift(nn.Module):
    def __init__(self, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        sources, batch, channels, length = wav.shape
        length = length - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                offsets = th.randint(
                    self.shift,
                    [1 if self.same else sources, batch, 1, 1], device=wav.device)
                offsets = offsets.expand(sources, -1, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav
