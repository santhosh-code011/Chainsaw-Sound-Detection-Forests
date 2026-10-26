"""
Chainsaw Sound Detection in Forests
Module: train.py
Synthetic Acoustic Benchmark Generator, PyTorch Training Engine, & Evaluation Matrix
"""

import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from model import (
    SAMPLE_RATE, DURATION, CLASSES, ChainsawDetectionCNN, 
    extract_mel_spectrogram, load_audio
)

def synthesize_forest_ambient(duration=DURATION, sr=SAMPLE_RATE):
    """Generate ambient forest sound: pink noise + wind gusts + bird whistles."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)
    white = np.random.randn(n)
    b, a = signal.butter(1, 0.05, btype='low')
    pink = signal.lfilter(b, a, white)
    pink = pink / (np.max(np.abs(pink)) + 1e-6)
    wind = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    audio = pink * wind * 0.4
    # Bird chirps
    for _ in range(np.random.randint(1, 3)):
        c_start = np.random.uniform(0.3, duration - 0.5)
        c_dur = np.random.uniform(0.15, 0.3)
        c_idx = (t >= c_start) & (t < c_start + c_dur)
        tc = t[c_idx] - c_start
        f = np.random.uniform(2500, 3600) + 800 * np.sin(2 * np.pi * 7.0 * tc)
        audio[c_idx] += 0.25 * np.sin(2 * np.pi * f * tc) * signal.windows.hann(len(tc))
    return (audio / (np.max(np.abs(audio)) + 1e-6)).astype(np.float32)

def synthesize_chainsaw_sound(duration=DURATION, sr=SAMPLE_RATE):
    """Generate 2-stroke chainsaw engine harmonic series + cutting blade friction."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)
    f0 = np.random.uniform(125.0, 165.0) * (1.0 + 0.12 * np.sin(2 * np.pi * 0.8 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    engine = sum((1.0 / (h ** 0.85)) * np.sin(h * phase + np.random.uniform(0, np.pi)) for h in range(1, 14))
    b, a = signal.butter(2, [1800 / (sr/2), 4800 / (sr/2)], btype='band')
    chain_noise = signal.lfilter(b, a, np.random.randn(n)) * 0.3
    combined = 0.75 * engine + 0.25 * chain_noise + 0.1 * synthesize_forest_ambient(duration, sr)
    return (combined / (np.max(np.abs(combined)) + 1e-6)).astype(np.float32)

def generate_benchmark_dataset(base_dir='data', train_n=50, test_n=15):
    """Generate structured WAV dataset."""
    print("Synthesizing acoustic dataset for Chainsaw Sound Detection...")
    demo_dir = os.path.join(base_dir, 'demo_samples')
    os.makedirs(demo_dir, exist_ok=True)
    splits = {'train': train_n, 'test': test_n}

    for split, count in splits.items():
        for cls_id, cls_name in enumerate(CLASSES):
            folder = os.path.join(base_dir, split, cls_name)
            os.makedirs(folder, exist_ok=True)
            gen = synthesize_forest_ambient if cls_name == 'Ambient_Forest' else synthesize_chainsaw_sound
            for i in range(count):
                wav_arr = (gen() * 32767).astype(np.int16)
                fpath = os.path.join(folder, f"{cls_name.lower()}_{i+1:03d}.wav")
                wav.write(fpath, SAMPLE_RATE, wav_arr)
                if split == 'test' and i == 0:
                    wav.write(os.path.join(demo_dir, f"{cls_name.lower()}_sample.wav"), SAMPLE_RATE, wav_arr)
    print("Dataset generated successfully!")

class AudioDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.fps = filepaths
        self.lbls = labels
    def __len__(self):
        return len(self.fps)
    def __getitem__(self, idx):
        y = load_audio(self.fps[idx])
        spec = extract_mel_spectrogram(y)
        return torch.tensor(spec).unsqueeze(0), torch.tensor(self.lbls[idx], dtype=torch.long)

def run_training(epochs=10, batch_size=16, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists('data/train'):
        generate_benchmark_dataset()

    def get_split(s):
        fps, lbls = [], []
        for i, c in enumerate(CLASSES):
            p = os.path.join('data', s, c)
            for f in os.listdir(p):
                if f.endswith('.wav'):
                    fps.append(os.path.join(p, f))
                    lbls.append(i)
        return fps, lbls

    tr_x, tr_y = get_split('train')
    te_x, te_y = get_split('test')
    train_loader = DataLoader(AudioDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(AudioDataset(te_x, te_y), batch_size=batch_size, shuffle=False)

    model = ChainsawDetectionCNN(num_classes=2).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    history = {'t_loss': [], 'v_acc': []}

    print("\nTraining ChainsawDetectionCNN on:", device)
    for ep in range(1, epochs + 1):
        model.train()
        r_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            r_loss += loss.item() * len(y)

        model.eval()
        c, tot = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                preds = model(x.to(device)).argmax(dim=1).cpu()
                c += (preds == y).sum().item()
                tot += len(y)
        acc = (c / tot) * 100.0
        sch.step()
        history['t_loss'].append(r_loss / len(tr_x))
        history['v_acc'].append(acc)
        print(f"Epoch {ep:02d} | Train Loss: {r_loss/len(tr_x):.4f} | Val Accuracy: {acc:.2f}%")

        if acc >= best_acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict(), 'best_acc': acc}, 'chainsaw_cnn.pt')

    print(f"\nModel trained! Checkpoint saved -> 'chainsaw_cnn.pt' (Peak Acc: {best_acc:.2f}%)")

    # Evaluate & Save Plots
    evaluate_model()

def evaluate_model():
    def get_test():
        fps, lbls = [], []
        for i, c in enumerate(CLASSES):
            p = os.path.join('data/test', c)
            for f in os.listdir(p):
                if f.endswith('.wav'):
                    fps.append(os.path.join(p, f))
                    lbls.append(i)
        return fps, lbls

    te_x, te_y = get_test()
    loader = DataLoader(AudioDataset(te_x, te_y), batch_size=16, shuffle=False)
    m = ChainsawDetectionCNN(num_classes=2)
    m.load_state_dict(torch.load('chainsaw_cnn.pt', map_location='cpu')['state_dict'])
    m.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(m(x).argmax(dim=1).numpy())
            trues.extend(y.numpy())

    print("\n" + "="*50)
    print("      CHAINSAW DETECTION - CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(trues, preds, target_names=CLASSES, digits=4))

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Chainsaw Detection Confusion Matrix')
    plt.xticks([0, 1], CLASSES, rotation=20)
    plt.yticks([0, 1], CLASSES)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200)
    plt.close()
    print("Confusion Matrix Saved -> 'confusion_matrix.png'\n")

if __name__ == '__main__':
    run_training()
