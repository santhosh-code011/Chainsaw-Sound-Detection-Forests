"""
Chainsaw Sound Detection in Forests
Module: model.py
Mel-Spectrogram Feature Extractor & Deep Residual Convolutional Neural Network
"""

import math
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 22050
DURATION = 3.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
F_MIN = 60.0
F_MAX = 10000.0

CLASSES = ['Ambient_Forest', 'Chainsaw_Activity']

def build_mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX):
    """Construct Slaney-normalized triangular Mel filterbank matrix."""
    n_freqs = 1 + n_fft // 2
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center > left:
            weights[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            weights[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights

MEL_FILTERBANK = build_mel_filterbank()

def load_audio(filepath, target_sr=SAMPLE_RATE, duration=DURATION):
    """Load WAV audio, convert to mono, resample, and normalize."""
    sr, y = wav.read(filepath)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    else:
        y = y.astype(np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != target_sr:
        num_samples = int(len(y) * target_sr / sr)
        y = signal.resample(y, num_samples)

    max_v = np.max(np.abs(y))
    if max_v > 1e-6:
        y = y / max_v

    target_samples = int(target_sr * duration)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
    return y.astype(np.float32)

def extract_mel_spectrogram(y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Extract decibel-scaled normalized Log Mel-Spectrogram (64 x time_frames)."""
    f, t, Zxx = signal.stft(
        y, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length,
        boundary=None, padded=False
    )
    power = np.abs(Zxx) ** 2
    mel = np.dot(MEL_FILTERBANK, power)
    mel = np.maximum(mel, 1e-10)
    mel_db = 10.0 * np.log10(mel)
    ref = np.max(mel_db)
    mel_db = np.clip(mel_db, ref - 80.0, ref)
    return ((mel_db - (ref - 80.0)) / 80.0).astype(np.float32)

class SqueezeExcitation(nn.Module):
    """Channel attention block for acoustic timbre recalibration."""
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(4, channels // r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, channels // r), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class ResidualConvBlock(nn.Module):
    """Residual convolutional block with SE attention."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.se = SqueezeExcitation(out_c)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.se(self.conv(x))
        return F.relu(out + res)

class ChainsawDetectionCNN(nn.Module):
    """
    Deep Residual CNN for Acoustic Chainsaw Sound Detection in Forests.
    Input: Log Mel-Spectrogram (1, 64, T) -> Binary Threat Classification (Ambient vs Chainsaw)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15)
        )
        self.stage2 = ResidualConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.2)

        self.stage3 = ResidualConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.drop2(self.pool2(self.stage2(x)))
        x = self.drop3(self.pool3(self.stage3(x)))
        x = self.stage4(x)
        return self.classifier(x)

    def predict_audio_waveform(self, waveform):
        self.eval()
        with torch.no_grad():
            spec = extract_mel_spectrogram(waveform)
            t = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits = self(t)
            probs = F.softmax(logits, dim=1).numpy()[0]
        return probs, spec
