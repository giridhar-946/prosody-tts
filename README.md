# Prosody-TTS

An End-to-End Speech Synthesis System with Prosody Control. Prosody-TTS is a data-efficient end-to-end speech synthesis model that combines the advantages of statistical parametric models and end-to-end neural network models. It also has a provision to modify or incorporate the desired prosody at the finer level by controlling the fundamental frequency (f0) and the phone duration.

This implementation is an extension of the Prosody-TTS to support multiple languages speakers. Vocos is used as the vocoder.

- Paper this is based on: Pamisetty & Murty, *Prosody-TTS: An End-to-End Speech Synthesis
  System with Prosody Control*, Circuits, Systems, and Signal Processing 42, 361–384 (2023),
  doi:[10.1007/s00034-022-02126-z](https://doi.org/10.1007/s00034-022-02126-z).
- What changed from the paper, and why, is listed in [docs/architecture.md](docs/architecture.md).

## Pipeline

```
Get phoneme level durations using the Montreal forced aligner (MFA)
                                                              ▼
 extract_features.py   wav ─► log-mel (Vocos), F0 (pYAAPT), TAPs (eGeMAPS) ─► exp/feats.arrow
                                                              ▼
 train.py              phones, durations (MFA), F0, TAPs, mel ─► Prosody-TTS ─► exp/weights/
                                                              ▼
 synthesize.py         text ─► espeak ─► phones ─► duration ─► F0/VAD ─► log-mel ─► Vocos ─► wav
```

## Installation

Tested with Python 3.10, PyTorch 2.11 (CUDA 12.8), and espeak-ng 1.51.

```bash
conda env create -f environment.yml          # or: see requirements.txt
conda activate prosodytts
sudo apt-get install espeak-ng                # phonemizer loads libespeak-ng
```

## Quick start

Full details: [data](docs/data.md), [training](docs/training.md), [inference](docs/inference.md).

```bash
# 1. environment (see Installation below)
conda activate prosodytts

# 2. features from an arrow dataset (docs/data.md)
python extract_features.py --arrow_dir /path/to/arrow_dir --exp_dir exp --workers 16

# 3. train
python train.py --arrow_dir /path/to/arrow_dir --exp_dir exp --frames_per_batch 2500

# 4. synthesize
python synthesize.py --exp_dir exp --speaker <speaker_key> \
    --text_file examples/sentences.txt --out_dir Synthesized
```

The vocoder checkpoint is not in this repository. Set `MLMS_VOCODER_DIR` (or pass
`--vocoder_dir`) to a folder containing `config.yaml` and `pytorch_model.bin` for a Vocos
model with 16 kHz audio, 1024 FFT, hop 320, 100 mel bins. The F5-TTS vocoder folder used
in development has exactly this layout.

## Repository layout

| File | Purpose |
|---|---|
| `extract_features.py` | Builds `exp/` from an arrow dataset: phone table, speakers, languages, `feats.arrow`, F0 stats, TAP scalers |
| `train.py` | Training loop, validation samples, checkpoints |
| `synthesize.py` | Text (or phones) to 16 kHz wav |
| `data.py` | Dataset, frame-budget sampler, collate, language and speaker keys |
| `model.py` | MLMS-TTS acoustic model (PyTorch) |
| `mel.py` | Vocos-compatible log-mel (20 ms, 16 kHz, 100 bins) |
| `vocoder.py` | Loads and runs the Vocos vocoder |
| `phonemize.py` | espeak-ng text to phone symbols, mapped to the training inventory |
| `utils.py` | Min-max helpers and forward/reverse phone position helpers |
| `tools/prepare_arrow_from_mfa.py` | Reference script that builds `raw.arrow` from MFA TextGrids (see its notes) |
| `tests/test_units.py` | Unit tests that run without data |
| `docs/` | Data, training, inference, architecture documentation |


## Documentation

- [docs/data.md](docs/data.md): data requirements, folder layout, arrow schema, alignment.
- [docs/training.md](docs/training.md): feature extraction, training, outputs, resuming.
- [docs/inference.md](docs/inference.md): synthesis, input formats, espeak voices.
- [docs/architecture.md](docs/architecture.md): model, losses, and differences from the paper.

## Citation

If you use this code, please cite the paper it is based on:

```bibtex
@article{pamisetty2023prosodytts,
  title   = {Prosody-TTS: An End-to-End Speech Synthesis System with Prosody Control},
  author  = {Pamisetty, Giridhar and Murty, K. Sri Rama},
  journal = {Circuits, Systems, and Signal Processing},
  volume  = {42},
  pages   = {361--384},
  year    = {2023},
  doi     = {10.1007/s00034-022-02126-z}
}
```

## Acknowledgements

Vocos (vocoder), F5-TTS (arrow data format and dynamic batching), espeak-ng and
`phonemizer` (phonemes), Montreal Forced Aligner (alignments), openSMILE eGeMAPSv02 (TAPs),
and `amfm_decompy` pYAAPT (F0).

## License

TBD (no license file yet).
