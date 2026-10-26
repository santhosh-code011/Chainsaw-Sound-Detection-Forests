"""
Chainsaw Sound Detection in Forests
Module: app.py
Interactive Dark Glassmorphic Web App & CLI Threat Predictor
"""

import os
import io
import sys
import base64
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from flask import Flask, request, jsonify, render_template_string

from model import (
    SAMPLE_RATE, DURATION, CLASSES, ChainsawDetectionCNN, 
    load_audio, extract_mel_spectrogram
)
from train import generate_benchmark_dataset

app = Flask(__name__)
MODEL_PATH = 'chainsaw_cnn.pt'
DEMO_DIR = 'data/demo_samples'

if not os.path.exists(DEMO_DIR):
    generate_benchmark_dataset()

model = ChainsawDetectionCNN(num_classes=2)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['state_dict'])
    model.eval()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chainsaw Sound Sentinel | Autonomous Forest Protection</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0a0f1d; --card: rgba(20, 28, 48, 0.8); --border: rgba(255,255,255,0.08); }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body { background: var(--bg); color: #f3f4f6; padding: 24px; min-height: 100vh; }
        .container { max-width: 1100px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 24px; }
        .grid { display: grid; grid-template-columns: 1fr 1.3fr; gap: 24px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 22px; backdrop-filter: blur(12px); }
        .btn-test { width: 100%; padding: 12px; margin-bottom: 10px; background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: #fff; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; }
        .btn-test:hover { background: rgba(245, 158, 11, 0.2); border-color: #f59e0b; }
        .upload-box { border: 2px dashed rgba(255,255,255,0.15); border-radius: 12px; padding: 24px; text-align: center; cursor: pointer; }
        .alert-banner { padding: 16px 20px; border-radius: 12px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; }
        .alert-safe { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }
        .alert-threat { background: rgba(239, 68, 68, 0.18); border: 1px solid #ef4444; color: #f87171; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
        .spec-box { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); margin-top: 14px; }
        .spec-box img { width: 100%; display: block; }
        audio { width: 100%; margin-top: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h2>🌲 Chainsaw Sound Detection in Forests</h2>
                <div style="font-size: 13px; color: #9ca3af;">Acoustic Deep Learning Mel-Spectrogram Residual CNN</div>
            </div>
            <div style="background: rgba(16,185,129,0.15); color: #34d399; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;">ACTIVE SENTINEL ONLINE</div>
        </header>

        <div class="grid">
            <div>
                <div class="card">
                    <h3 style="margin-bottom: 14px;">1-Click Audio Test Presets</h3>
                    <button class="btn-test" onclick="runSample('chainsaw')">🪚 Test Chainsaw Activity (Logging Threat)</button>
                    <button class="btn-test" onclick="runSample('ambient_forest')">🍃 Test Ambient Forest (Safe Ecosystem)</button>

                    <h3 style="margin: 20px 0 12px 0;">Upload Custom Audio (.wav)</h3>
                    <div class="upload-box" onclick="document.getElementById('f').click()">
                        <div>📂 Click to upload WAV file</div>
                    </div>
                    <input type="file" id="f" accept=".wav" style="display:none;" onchange="uploadAudio(event)">
                    <audio id="player" controls style="display:none;"></audio>
                </div>
            </div>

            <div>
                <div class="card">
                    <h3 style="margin-bottom: 14px;">Real-Time Acoustic Threat Assessment</h3>
                    <div id="banner" class="alert-banner alert-safe">
                        <div>
                            <div id="statusTitle" style="font-size: 18px; font-weight: 700;">AMBIENT FOREST</div>
                            <div id="statusDesc" style="font-size: 12px; opacity: 0.85;">Ecosystem clear. No motorized logging detected.</div>
                        </div>
                        <div id="conf" style="font-size: 20px; font-weight: 700;">99.2%</div>
                    </div>

                    <h4 style="margin-top: 10px;">Log Mel-Spectrogram Heatmap</h4>
                    <div class="spec-box">
                        <img id="specImg" src="" alt="Mel Spectrogram">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runSample(type) {
            const res = await fetch(`/api/sample/${type}`);
            const data = await res.json();
            render(data);
        }

        async function uploadAudio(e) {
            const file = e.target.files[0];
            if (!file) return;
            const fd = new FormData();
            fd.append('audio', file);
            const res = await fetch('/api/predict', { method: 'POST', body: fd });
            const data = await res.json();
            render(data);
        }

        function render(data) {
            const b = document.getElementById('banner');
            const t = document.getElementById('statusTitle');
            const d = document.getElementById('statusDesc');
            const c = document.getElementById('conf');

            c.innerText = (data.confidence * 100).toFixed(1) + '%';
            if (data.class === 'Chainsaw_Activity') {
                b.className = 'alert-banner alert-threat';
                t.innerText = '🚨 CHAINSAW LOGGING DETECTED!';
                d.innerText = 'Illegal deforestation acoustic harmonic frequency detected!';
            } else {
                b.className = 'alert-banner alert-safe';
                t.innerText = '✅ FOREST ECOSYSTEM SECURE';
                d.innerText = 'Normal environmental canopy acoustics.';
            }

            document.getElementById('specImg').src = 'data:image/png;base64,' + data.spec_b64;
            const p = document.getElementById('player');
            p.src = 'data:audio/wav;base64,' + data.audio_b64;
            p.style.display = 'block';
            p.play().catch(()=>{});
        }
        window.onload = () => runSample('chainsaw');
    </script>
</body>
</html>
"""

def spec_to_b64(spec):
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=140)
    fig.patch.set_facecolor('#0d1322')
    ax.set_facecolor('#0d1322')
    im = ax.imshow(spec, origin='lower', aspect='auto', cmap='magma')
    ax.axis('off')
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/api/sample/<name>')
def get_sample(name):
    fname = f"{name.lower()}_sample.wav"
    fp = os.path.join(DEMO_DIR, fname)
    if not os.path.exists(fp):
        fp = os.path.join(DEMO_DIR, f"{name.lower()}_activity_sample.wav")
    y = load_audio(fp)
    probs, spec = model.predict_audio_waveform(y)
    idx = int(np.argmax(probs))
    with open(fp, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    return jsonify({
        'class': CLASSES[idx],
        'confidence': float(probs[idx]),
        'spec_b64': spec_to_b64(spec),
        'audio_b64': audio_b64
    })

@app.route('/api/predict', methods=['POST'])
def predict_upload():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['audio']
    tmp = 'temp.wav'
    f.save(tmp)
    try:
        y = load_audio(tmp)
        probs, spec = model.predict_audio_waveform(y)
        idx = int(np.argmax(probs))
        with open(tmp, 'rb') as af:
            audio_b64 = base64.b64encode(af.read()).decode('utf-8')
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return jsonify({
        'class': CLASSES[idx],
        'confidence': float(probs[idx]),
        'spec_b64': spec_to_b64(spec),
        'audio_b64': audio_b64
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=str, help='WAV file to predict')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    if args.predict:
        y = load_audio(args.predict)
        probs, _ = model.predict_audio_waveform(y)
        idx = int(np.argmax(probs))
        print(f"\nResult: {CLASSES[idx]} (Confidence: {probs[idx]*100:.2f}%)\n")
    else:
        print(f"\n* Dashboard online on http://127.0.0.1:{args.port}\n")
        app.run(host='0.0.0.0', port=args.port, debug=False)
