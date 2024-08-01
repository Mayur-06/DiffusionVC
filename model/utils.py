# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch
import torchaudio
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import torchaudio.transforms as T

from model.base import BaseModule


def mse_loss(x, y, mask, n_feats):
    loss = torch.sum(((x - y)**2) * mask)
    return loss / (torch.sum(mask) * n_feats)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


class PseudoInversion(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft):
        super(PseudoInversion, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=8000)
        mel_basis_inverse = np.linalg.pinv(mel_basis)
        mel_basis_inverse = torch.from_numpy(mel_basis_inverse).float()
        self.register_buffer("mel_basis_inverse", mel_basis_inverse)

    def forward(self, log_mel_spectrogram):
        mel_spectrogram = torch.exp(log_mel_spectrogram)
        stftm = torch.matmul(self.mel_basis_inverse, mel_spectrogram)
        return stftm


class InitialReconstruction(BaseModule):
    def __init__(self, n_fft, hop_size):
        super(InitialReconstruction, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)
        #self.istft_transform = T.InverseSTFT(n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.n_fft, window=window, center=True)

    def forward(self, stftm):
        real_part = torch.ones_like(stftm, device=stftm.device)
        imag_part = torch.zeros_like(stftm, device=stftm.device)
        stft = torch.stack([real_part, imag_part], -1)*stftm.unsqueeze(-1)
        # Convert to complex tensor
        stft = torch.view_as_complex(stft)
        istft = torch.istft(stft, n_fft=self.n_fft, 
                           hop_length=self.hop_size, win_length=self.n_fft, 
                           window=self.window, center=True)
        # # Convert to complex tensor
        # stft = torch.view_as_complex(stft)
        #  # Apply Inverse STFT
        # #istft = self.istft_transform(stft)
        return istft.unsqueeze(1)


# Fast Griffin-Lim algorithm as a PyTorch module
class FastGL(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft, hop_size, momentum=0.99):
        super(FastGL, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.momentum = momentum
        self.pi = PseudoInversion(n_mels, sampling_rate, n_fft)
        self.ir = InitialReconstruction(n_fft, hop_size)
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)
        #self.istft_transform = T.InverseSTFT(n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.n_fft, window=window, center=True)
    # @torch.no_grad()
    # def forward(self, s, n_iters=32):
    #     c = self.pi(s)
    #     x = self.ir(c)
    #     x = x.squeeze(1)
    #     c = c.unsqueeze(-1)

    #     prev_angles = torch.zeros_like(c, device=c.device)
    #     for _ in range(n_iters):        
    #         s = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_size, 
    #                        win_length=self.n_fft, window=self.window, 
    #                        center=True, return_complex=True)
    #         #real_part, imag_part = s.unbind(-1)

    #         # Compute magnitude and angles from the complex tensor
    #         #stftm = torch.abs(s)
    #         angles = torch.angle(s)
    #         #angles = angles.unsqueeze(-1)
    #         #m = self.momentum
    #         print(f"angles shape: {angles.shape}")
    #         print(f"prev_angles shape: {prev_angles.shape}")
    #         print(f"momentum shape: {c.shape}")


    #         #stftm = torch.sqrt(torch.clamp(real_part**2 + imag_part**2, min=1e-8))
    #         #angles = s / stftm.unsqueeze(-1)
    #         s = c * (angles + self.momentum * (angles - prev_angles))

    #         # Convert to complex tensor
    #         #s = torch.view_as_complex(s)
    #         a=torch.abs(s) 
    #         # angles = angles.squeeze(-1)
    #         print(f"ashape: {a.shape}")
    #         print(f"angles shape: {angles.shape}")
    #         #print(f"momentum shape: {c.shape}")
    #         s = torch.polar(a, angles)
    #         print(f"s shape: {s.shape}")
    #         x = torch.istft(s, n_fft=self.n_fft, hop_length=self.hop_size, 
    #                                         win_length=self.n_fft, window=self.window, 
    #                                         center=True)
    #         prev_angles = angles
    #     return x.unsqueeze(1)
    @torch.no_grad()
    def forward(self, s, n_iters=32):
        c = self.pi(s)
        x = self.ir(c)
        x = x.squeeze(1)
        c = c.unsqueeze(-1)

        prev_angles = torch.zeros_like(c, device=c.device)
        for _ in range(n_iters):        
            s = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_size, 
                           win_length=self.n_fft, window=self.window, 
                           center=True, return_complex=True)
            
            angles = torch.angle(s)
            angles = angles.unsqueeze(-1)

            print(f"angles shape: {angles.shape}")
            print(f"prev_angles shape: {prev_angles.shape}")
            print(f"momentum shape: {c.shape}")

            s = c * (angles + self.momentum * (angles - prev_angles))

            # Remove the extra dimension before calling torch.polar
            angles = angles.squeeze(-1)
            prev_angles = prev_angles.squeeze(-1)
            a = torch.abs(s)
            a=a.squeeze(-1)

            print(f"ashape: {a.shape}")
            print(f"angles shape: {angles.shape}")

            s = torch.polar(a, angles)
            print(f"s shape: {s.shape}")

            x = torch.istft(s, n_fft=self.n_fft, hop_length=self.hop_size, 
                                            win_length=self.n_fft, window=self.window, 
                                            center=True)
            prev_angles = angles.unsqueeze(-1)
        return x.unsqueeze(1)