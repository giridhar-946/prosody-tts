"""Log-mel spectrogram matching the Vocos vocoder used by F5-TTS.

Copied from f5_tts/model_v5/modules.py (get_vocos_mel_spectrogram) with the
Vocos 20 ms configuration: 16 kHz, n_fft 1024, hop 320 (20 ms), 100 mel bins.
Output is the natural log of the power-1 mel magnitude, clamped at 1e-5, with
no further normalisation, which is the domain Vocos decodes from.

Checked against vocos.feature_extractors.MelSpectrogramFeatures with the
f5_tts/vocoder/config.yaml settings: identical output (max abs diff 0.0).
"""
import torchaudio

SAMPLE_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 320          # 20 ms at 16 kHz
WIN_LENGTH = 1024
N_MEL_CHANNELS = 100
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH   # 50 frames per second


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=N_FFT,
    n_mel_channels=N_MEL_CHANNELS,
    target_sample_rate=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
):
    """waveform: (1, n_samples) or (n_samples,) float tensor at target_sample_rate.
    Returns (n_mel_channels, n_frames) log-mel. n_frames = n_samples // hop + 1."""
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel.squeeze(0)


def mel_frames(n_samples, hop_length=HOP_LENGTH):
    """Number of mel frames produced for n_samples with center=True padding."""
    return n_samples // hop_length + 1
