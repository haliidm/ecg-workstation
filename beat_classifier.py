"""Adapter for the local ECG project's single-lead beat classifier."""

import os
from pathlib import Path
import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np


class BeatClassifier:
    def __init__(self):
        bundled = Path(__file__).resolve().parent / 'vendor' / 'beat'
        default = bundled if bundled.is_dir() else Path('D:/FRX-BCG/ECG')
        self.root = Path(os.environ.get('ECG_BEAT_MODEL_ROOT', str(default)))
        self.model = None

    def predict(self, signal, fs):
        checkpoint = self.root / 'checkpoints' / 'ecg_cnn_transformer_16s.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'找不到心搏模型权重：{checkpoint}')
        if str(self.root) not in sys.path:
            sys.path.append(str(self.root))

        import torch
        from scipy.signal import resample_poly
        from wfdb.processing import xqrs_detect
        from mitbih_aami.cnn_transformer import MultiScaleCNNTransformer
        from mitbih_aami.ecg_data import _filter_and_normalize
        from mitbih_aami.label_config import CLASS_NAMES, FS, WINDOW_SAMPLES
        from mitbih_aami.record_inference_metrics import predict_record

        peaks = xqrs_detect(signal, fs=fs, verbose=False)
        if not peaks.size:
            return []
        if self.model is None:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = MultiScaleCNNTransformer(input_samples=int(state.get('input_samples', WINDOW_SAMPLES)))
            model.load_state_dict(state['model_state'])
            self.model = model.eval()

        ratio = Fraction(FS / fs).limit_denominator(10000)
        resampled = resample_poly(signal, ratio.numerator, ratio.denominator)
        model_signal = _filter_and_normalize(resampled, FS)
        logits = predict_record(self.model, SimpleNamespace(signal=model_signal), 'cpu')
        radius = round(0.05 * FS)
        labels = ('UNKNOWN', *CLASS_NAMES[1:])
        beats = []
        for peak in peaks:
            center = min(len(model_signal) - 1, round(peak * FS / fs))
            scores = logits[:, max(0, center - radius):center + radius + 1].mean(axis=1)
            beats.append((int(peak), labels[int(np.argmax(scores))]))
        return beats
