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
conda config --env --add channels conda-forge
conda install montreal-forced-aligner
```

## Quick start

Full details: [data](docs/data.md), [training](docs/training.md), [inference](docs/inference.md).

```bash
# 1. environment (see Installation below)
conda activate prosodytts

# 2. Preprocess data. See docs/data.md for data structure
python tools/preprocess_data.py --data_path /path/to/data

# 3. Prepare pronunciation dictionary. If working with multiple languages, repeat the same for all languages
python prepare_dict.py --language <langauge1> --path /path/to/data
python prepare_dict.py --language <langauge2> --path /path/to/data

# 4. Align speech and phonemes to get phoneme durations.
mfa validate --temporary_directory /path/to/tmp/dir --clean --no_final_clean --ignore_acoustics --num_jobs 24
       --use_mp /path/to/data/language /path/to/data/language/language.txt
python tools/move_oov.py --oov_path ~align_tmp/language/utterance_oovs.txt --data_path /path/to/data/languge --tmp_path temp_dir
python tools/move_missing.py --missing_path ~align_tmp/language/missing_utterances.txt --data_path /path/to/data/languge --tmp_path temp_dir
Set the frame shift in configs/config_mfa.yaml
mfa train --config_path configs/config_mfa.yaml --temporary_directory ~/path/to/align-tmp-dir/ --clean --no_final_clean --num_jobs 24 --use_mp /path/to/data/language /path/to/data/language.txt /path/to/data/language/acoustic_model.zip --output_directory /path/to/data/language

Repeat the step for all the languages if training for multiple speakers

# 5. Extract duration data from the MFA textgrids.
python tools/prepare_align_data.py --exp_dir /path/to/exp/dir --mfa_dir_list /comma/separated/list/of/mfa/out/dirs

# 6. Prepare training data
python prepare_train_data.py --exp_dir /path/to/exp/dir --workers 16
## If error occurs with iso369, uninstall python-iso639 in the local by deactivating the prosodytts env (pip uninstall  python-iso639)

# 7. train
Download the Vocos checkpoint from https://drive.google.com/file/d/1LJ6HnkWk0RvfBVPxYeeTAJOqOfHugamL/view?usp=sharing and place it in the vocoder folder.
CUDA_VISIBLE_DEVICES=6 python train.py --exp_dir /path/to/exp/dir --frames_per_batch 2500

# 8. synthesize
python synthesize.py --exp_dir /path/to/exp/dir --speaker <speaker_key> \
    --text_file examples/sentences.txt --out_dir Synthesized
```


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
