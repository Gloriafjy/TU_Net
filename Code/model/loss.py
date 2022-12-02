import torch
from scipy import linalg
import numpy as np
import scipy
import librosa
from model import pmsqe_com

EPS = 1e-8


class mse_loss(object):
    def __call__(self, outputs, labels, loss_mask):
        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        loss = torch.sum((masked_outputs - masked_labels) ** 2.0) / torch.sum(loss_mask)
        return loss


class TorchSignalToFrames(object):
    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = (sig_len // self.frame_shift)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class stftm_loss(object):
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.frame = TorchSignalToFrames(frame_size=self.frame_size,
                                         frame_shift=self.frame_shift)
        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR = np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().cuda()
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().cuda()
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().cuda()

    def __call__(self, outputs, labels, loss_mask):
        outputs = self.frame(outputs)
        labels = self.frame(labels)
        loss_mask = self.frame(loss_mask)
        outputs = self.get_stftm(outputs)
        labels = self.get_stftm(labels)

        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        if self.loss_type == 'mse':
            loss = torch.sum((masked_outputs - masked_labels) ** 2) / torch.sum(loss_mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(masked_outputs - masked_labels)) / torch.sum(loss_mask)

        return loss

    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        stftm = torch.abs(stft_R) + torch.abs(stft_I)
        return stftm


class perceptual_loss(object):
    def __init__(self, frame_size=512, frame_shift=256):
        self.compute_pmsqe = pmsqe_com.PMSQE(window_name='hamming')

    def __call__(self, outputs_3, labels_3, loss_mask):
        total = 0
        for i in range(outputs_3.shape[0]):
            outputs, labels = np.array(outputs_3[i][0]), np.array(labels_3[i][0])
            outputs = self.overlapped_windowing(outputs)
            labels = self.overlapped_windowing(labels)
            outputs = self.squared_magnitude_computation(outputs)
            labels = self.squared_magnitude_computation(labels)
            outputs, labels = torch.from_numpy(outputs), torch.from_numpy(labels)
            loss_mask = torch.ones(labels.shape)
            masked_outputs = outputs * loss_mask
            masked_labels = labels * loss_mask
            pad_mask = torch.ones(masked_outputs.shape)
            wD_frame, wDA_frame = self.compute_pmsqe(masked_outputs, masked_labels, pad_mask)
            loss = torch.mean(0.1 * (wD_frame + 0.309 * wDA_frame))
            total = total + loss
        mean = total / (outputs_3.shape[0])
        return mean

    def overlapped_windowing(self, x, frame_length=512, hop_length=256):
        overlapped_slices = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
        window = scipy.signal.hamming(frame_length, sym=False)
        window = window.reshape((-1, 1))
        return overlapped_slices * window

    def squared_magnitude_computation(self, frames):
        spectrum = np.square(np.abs(np.fft.rfft(frames.T)))
        return spectrum


def calc_sdr_torch(estimation, origin, mask=None):
    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask

    origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + EPS
    scale = torch.sum(origin * estimation, 1, keepdim=True) / origin_power
    est_true = scale * origin
    est_res = estimation - est_true
    true_power = torch.pow(est_true, 2).sum(1)
    res_power = torch.pow(est_res, 2).sum(1)
    loss = - (10 * torch.log10(true_power) - 10 * torch.log10(res_power))
    return loss.mean()
