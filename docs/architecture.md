# Architecture

The model is the Prosody-TTS design (Pamisetty & Murty, 2023) with the changes listed
below. The PyTorch code is a layer-by-layer port of an earlier TensorFlow/Keras graph,
which explains some of the layer choices (for example the windowed attention blocks).

## Overview

```
phones ──► phone embedding (128, frozen) ──┐
language (16, frozen) ──► text encoder ────┤
                                           ▼
                       duration head (Dense) ──► phone durations (20 ms frames)
                                           │
                 upsample by durations (gather_frames)
                                           │
              ┌────────────────────────────┴───────────────────────────┐
              ▼                                                        ▼
  VAD/F0 estimator ──► VAD (sigmoid), pitch (F0, min-max) ──┐     (encoder + speaker (32))
  frame position (fwd/rev 16+16)                            │            │
                                                            ▼            ▼
                              F0 embedding (32) ◄── reference F0 ── concat ──► acoustic decoder
                                                                            │
                                                              new acoustic decoder (attention)
                                                                            │
                                                                 mel head (linear, 100 bins)
                                                                            │
                                                          TAPs decoder ──► TAPs (25, auxiliary)
                                                                            │
                                                          log-mel ──► Vocos (frozen) ──► wave
```

## Components

Shapes are batch-first, `B` utterances, `L` phones, `F` frames (20 ms).

| Block | Input → output | Details |
|---|---|---|
| Phone embedding | `(B, L)` → `(B, L, 128)` | `nn.Embedding`, frozen, truncated-normal init (σ = 0.5) |
| Language embedding | `(B, L)` → `(B, L, 16)` | frozen |
| Speaker embedding | `(B, L)` → `(B, L, 32)` | frozen |
| Text encoder | `(B, L, 144)` → `(B, L, 128)` | 8 Conv1D filters, kernels 1…8, ReLU, BatchNorm; max-pool (2, stride 1); Conv1D k=3 + BN; Conv1D k=3 + BN; residual add of the phone embedding; BiGRU (2 × 64) |
| Duration head | `(B, L, 160)` → `(B, L, 1)` | Dense 64 → Dense 64 → Dense 1 (ReLU) |
| Frame position | `(B, F)` → `(B, F, 16)` × 2 | sinusoidal, frozen (forward and reverse position inside each phone) |
| VAD/F0 estimator | `(B, F, 160)` or `(B, F, 192)` → `(B, F, 64)` | Dense 128; 2 blocks of windowed self-attention (4 heads × 32, context ±10 frames), BatchNorm, two Dense 128 layers per block |
| VAD head | `(B, F, 64)` → `(B, F, 1)` | Dense 64 → Dense 1 (sigmoid) |
| Pitch head | `(B, F, 64)` → `(B, F, 1)` | Dense 64 → Dense 1 (ReLU), multiplied by VAD |
| F0 embedding | reference F0 (Hz, clipped to 0–399) → `(B, F, 32)` | sinusoidal table, frozen |
| Acoustic decoder | `(B, F, 224)` → `(B, F, 128)` | 10 Conv1D filters, kernels 2…11, ReLU, BN; max-pool; 3 × Conv1D k=3 + BN; BiGRU (2 × 64) |
| New acoustic decoder | `(B, F, 128)` → `(B, F, 128)` | 3 blocks; each block computes one attention output and reuses it in five residual sub-steps (BN, Dense 128 × 2) |
| Mel head | `(B, F, 128)` → `(B, F, 100)` | Dense, **linear** output (log-mel values are negative) |
| TAPs decoder | `(B, F, 100)` → `(B, F, 25)` | 4 × Conv1D k=5 + BN; Dense 64; two Dense 25 heads (value × ReLU gate) |

All frame-level outputs are multiplied by the frame mask, so padding stays at zero.

## Losses

Training minimises the weighted sum

```
L = 20·L_dur + 10·L_pitch + 1·L_vad + 10·L_mel + 1·L_taps + L2
```

| Term | Target | Loss |
|---|---|---|
| `L_dur` | MFA phone duration, min-max to 1–50 frames | L1 |
| `L_pitch` | F0, min-max per speaker (voiced range from data) | L1 |
| `L_vad` | voiced flag (F0 > 10 Hz) | binary cross-entropy |
| `L_mel` | Vocos log-mel (natural log, clamped at 1e-5) | L1 |
| `L_taps` | eGeMAPS TAPs, MinMax per speaker | MSE |
| `L2` | attention weights, coefficient 1e-5 | squared norm |

Adam with learning rate 1e-3; ReduceLROnPlateau (factor 0.1, patience 30, minimum 1e-8);
early stopping on validation loss.

## Feature pipeline

| Feature | Source | Frame rate |
|---|---|---|
| Mel | torchaudio MelSpectrogram, 16 kHz, FFT 1024, window 1024, hop 320, 100 mel bins, power 1, `log(clamp(·, 1e-5))` | 20 ms |
| F0 | pYAAPT (`amfm_decompy`), 50–400 Hz, frame space 20 ms; unvoiced set to 1 Hz | 20 ms |
| TAPs | openSMILE eGeMAPSv02 LLDs (25 values) | 10 ms, mapped to 20 ms |
| Phone durations | MFA alignments, in 20 ms frames | 20 ms |

The mel implementation matches Vocos's own feature extractor for the same settings (checked
with a maximum absolute difference of 0.0).

## Differences from the paper

| Aspect | Paper (Prosody-TTS, 2023) | This system |
|---|---|---|
| Sampling and frame rate | 16 kHz, 5 ms frame shift, 25 ms frame length | 16 kHz, 20 ms hop (320 samples) |
| Phones | IPA from a text analyser, 58 phones (Telugu) | MFA phone alignments (arrow data); espeak-ng for text (see [data.md](data.md#6-phones-and-espeak)) |
| Phone embedding | fixed 128-d, non-trainable | same (128-d, frozen), plus frozen language (16) and speaker (32) embeddings |
| Text encoder | Conv1D bank (k = 1…8) + 2 projections + BiGRU | same structure, plus language input and BatchNorm |
| Duration | Dense + ReLU, MAPE loss | 3-layer Dense, L1 loss on min-max-scaled durations; targets are MFA durations |
| Frame-level F0 | 3 Conv1D projections (k = 5), frame position encoding (16 + 16), F0 and BAP outputs, MSE on log-F0 | attention-based estimator with the same frame position encoding; F0 min-max per speaker, L1 loss; no BAP |
| Band aperiodicity (BAP) | predicted and used for the excitation | **not used**. Replaced by eGeMAPS TAPs (25 values) as an auxiliary target |
| Acoustic decoder | Conv1D bank (k = 1…16) + 3 projections + BiGRU; 80-bin mel, ReLU output, MSE on log-mel | Conv1D bank (k = 2…11) + BiGRU, then attention blocks; 100-bin log-mel with a **linear** output, L1 loss |
| Excitation | sawtooth pulse train and BAP-scaled noise, fed to the vocoder | **not used**. The vocoder conditions on log-mel only |
| Vocoder | Waffler-style CNN, trained jointly with the acoustic model (time-domain and multi-resolution STFT losses) | **Vocos**, pretrained and frozen (16 kHz, hop 320, 100 bins) |
| Training data | IndicTTS Telugu, one male speaker, about 4 hours | arrow datasets built from MFA alignments; multi-speaker |
| Batching | batch size 16 | frame-budget batches (default 2500 frames, at most 32 utterances) |
| Prosody control | duration and F0 scaling (Section 4) | not implemented in the inference script |
| Evaluation | MOS, RTF, GPE/VDE/FFE, duration RMSE/MAE/PCC | not included yet |

The paper's loss sum `L_total = L_dur + L_acst + L_vocoder` does not apply here, because the
vocoder is frozen and not trained with the acoustic model.

## Known gaps

- The duration and F0 controls described in Section 4 of the paper are not exposed.
- Without BAP, the model does not separate aperiodic energy. The vocoder must infer voicing
  from the log-mel and the VAD-shaped F0.
- The language and speaker embeddings are frozen. A new speaker needs a new row, so
  speakers must be known in `spk2id.json` before training.
- Keras mask propagation through the BiGRUs is not reproduced. Padding frames enter the
  recurrent layers.
