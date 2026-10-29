"""Vocos vocoder loading and decoding, as used by F5-TTS (infer/utils_infer_v5.py).

Default checkpoint: /data/AccentConv/Giridhar/f5_tts/vocoder
    config.yaml          Vocos hyper-parameters (16 kHz, hop 320, 100 mel bins)
    pytorch_model.bin    weights
"""
import os

import torch
from vocos import Vocos

DEFAULT_VOCODER_DIR = os.environ.get('MLMS_VOCODER_DIR', '/data/AccentConv/Giridhar/f5_tts/vocoder')
SAMPLE_RATE = 16_000


def load_vocos(vocoder_dir=DEFAULT_VOCODER_DIR, device='cpu'):
    config_path = os.path.join(vocoder_dir, 'config.yaml')
    model_path = os.path.join(vocoder_dir, 'pytorch_model.bin')
    model = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f'vocos load: {len(missing)} missing, {len(unexpected)} unexpected keys')
    return model.eval().to(device)


@torch.no_grad()
def decode(vocos, log_mel):
    """log_mel: (n_mel, T) or (1, n_mel, T) float tensor -> waveform (N,) at 16 kHz."""
    if log_mel.dim() == 2:
        log_mel = log_mel.unsqueeze(0)
    audio = vocos.decode(log_mel.to(next(vocos.parameters()).device))
    return audio.squeeze(0).cpu()
