"""Build the multi-speaker experiment directory from the F5-TTS arrow files.

Reads (reused unchanged from f5_tts):
    <exp_dir>/raw.arrow     audio_path, text, duration, phn_durs
    <exp_dir>/vocab.txt     phone inventory of the arrow files, one symbol per line
    <exp_dir>/spk2id.json   speaker id map (kept, so ids match the F5 data)

Writes into <exp_dir>/:
    phones.txt            'symbol<TAB>id', id 0 = <eps> (padding)
    speakers.txt                speaker keys in id order (line number = id)
    languages.txt               accent-group names (see data.language_of)
    feats.arrow                 per utterance: raw_index, num_frames, f0 (Hz, 20 ms grid),
                                    taps (num_frames x 25, 20 ms grid), raw from eGeMAPS at 10 ms
    speaker_stats.json          per speaker min_f0 / max_f0 over voiced frames
    scalers/minmax.scaler_<spk> per speaker MinMaxScaler over TAP frames

Per utterance (computed in parallel):
    mel frames  T = n_samples // 320 + 1 at 16 kHz (mel.py)
    F0          pYAAPT, frame_space 20 ms, mapped onto the T mel frames
    TAPs        openSMILE eGeMAPSv02 LLDs (10 ms), mapped onto the T mel frames

Usage:
    python extract_features.py --exp_dir /data/AccentConv/Giridhar/f5_tts/data/Model_V5_Comb_char \
        --exp_dir exp --workers 64 --max_utts_per_speaker 300
"""
import os

# one thread per process: the pool provides the parallelism
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import json
import multiprocessing as mp
import random
import warnings
from collections import defaultdict

import joblib
import numpy as np
from amfm_decompy import basic_tools, pYAAPT
from datasets import Dataset as HFDataset
from datasets.arrow_writer import ArrowWriter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from data import (MAX_DURATION, MIN_DURATION, N_TAPS, language_of, load_wav_16k, proportional_index, speaker_of)
from mel import SAMPLE_RATE, get_vocos_mel_spectrogram

F0_MIN, F0_MAX = 50.0, 400.0          # no gender in the arrow metadata; wide range for all
F0_FRAME_SPACE_MS = 20.0              # matches the 20 ms mel grid
DEFAULT_F0 = {'min_f0': F0_MIN, 'max_f0': F0_MAX}

_smile = None


def _init_worker():
    global _smile
    import opensmile
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore', category=RuntimeWarning)  # pYAAPT divide-by-zero on silence
    _smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                             feature_level=opensmile.FeatureLevel.LowLevelDescriptors)


def extract_one(job):
    try:
        return _extract(job)
    except Exception as e:  # one bad wav must not stop a long run
        print(f'skipped {job[1]}: {e}', flush=True)
        return None


def _extract(job):
    raw_index, path = job
    wav = load_wav_16k(path)
    T = get_vocos_mel_spectrogram(wav).shape[-1]
    samples = wav.numpy().astype(np.float64)

    signal = basic_tools.SignalObj(data=samples, fs=SAMPLE_RATE)
    pitch = pYAAPT.yaapt(signal, **{'f0_min': F0_MIN, 'f0_max': F0_MAX,
                                    'frame_length': 25.0, 'frame_space': F0_FRAME_SPACE_MS})
    f0 = np.asarray(pitch.samp_values, dtype=np.float32)
    f0[f0 < 1] = 1
    f0 = f0[proportional_index(len(f0), T)]

    taps = _smile.process_signal(samples.astype(np.float32), SAMPLE_RATE).to_numpy().astype(np.float32)
    taps = taps[proportional_index(len(taps), T)]
    assert taps.shape == (T, N_TAPS), (path, taps.shape, T)
    return {'raw_index': raw_index, 'num_frames': T, 'f0': f0, 'taps': taps}


def select_rows(raw_ds, max_per_speaker, seed):
    """Indices of utterances in the usable duration range, optionally sub-sampled per speaker."""
    durations = raw_ds['duration']
    paths = raw_ds['audio_path']
    candidates = [i for i, d in enumerate(durations) if MIN_DURATION <= d <= MAX_DURATION]
    if max_per_speaker <= 0:
        return candidates
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for i in candidates:
        by_spk[speaker_of(paths[i])].append(i)
    selected = []
    for spk in sorted(by_spk):
        idx = by_spk[spk]
        selected.extend(rng.sample(idx, min(max_per_speaker, len(idx))))
    return sorted(selected)


def write_vocab_and_speakers(exp_dir, raw_paths, languages_all):

    with open(os.path.join(exp_dir, 'vocab.txt'), encoding='utf-8') as f:
        vocab = f.read().split('\n')
    if vocab and vocab[-1] == '':
        vocab = vocab[:-1]
    with open(os.path.join(exp_dir, 'phones.txt'), 'w', encoding='utf-8') as f:
        f.write('<eps>\t0\n')
        for k, sym in enumerate(vocab, start=1):
            f.write(f'{sym}\t{k}\n')

    with open(os.path.join(exp_dir, 'spk2id.json'), encoding='utf-8') as f:
        spk2id = json.load(f)
    ordered = sorted(spk2id, key=lambda s: spk2id[s])
    if [spk2id[s] for s in ordered] != list(range(len(ordered))):
        raise SystemExit('spk2id.json ids must be 0..N-1')
    missing = {speaker_of(p) for p in raw_paths} - set(spk2id)
    if missing:
        raise SystemExit(f'speakers missing from spk2id.json: {sorted(missing)[:10]}')
    with open(os.path.join(exp_dir, 'speakers.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(ordered) + '\n')

    languages = sorted(languages_all)
    with open(os.path.join(exp_dir, 'languages.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(languages) + '\n')
    return len(vocab), len(ordered), languages


def compute_speaker_tables(exp_dir, raw_paths):
    """Per speaker F0 range and TAP scaler from feats.arrow."""
    feats = HFDataset.from_file(os.path.join(exp_dir, 'feats.arrow'))
    f0_min, f0_max = {}, {}
    scalers = defaultdict(MinMaxScaler)
    for row in tqdm(feats, desc='Speaker stats'):
        spk = speaker_of(raw_paths[row['raw_index']])
        f0 = np.asarray(row['f0'], dtype=np.float32)
        voiced = f0[f0 > 1]
        if voiced.size:
            f0_min[spk] = min(f0_min.get(spk, np.inf), float(voiced.min()))
            f0_max[spk] = max(f0_max.get(spk, -np.inf), float(voiced.max()))
        scalers[spk].partial_fit(np.asarray(row['taps'], dtype=np.float32).reshape(-1, N_TAPS))

    stats = {}
    for spk in scalers:
        stats[spk] = {'min_f0': f0_min.get(spk, DEFAULT_F0['min_f0']),
                      'max_f0': f0_max.get(spk, DEFAULT_F0['max_f0'])}
        joblib.dump(scalers[spk], os.path.join(exp_dir, 'scalers', f'minmax.scaler_{spk}'))
    with open(os.path.join(exp_dir, 'speaker_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=1)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True, help='Experiment directory, e.g. exp_data/check')
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 4))
    p.add_argument('--max_utts_per_speaker', type=int, default=0, help='0 = all usable utterances')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--chunksize', type=int, default=8)
    args = p.parse_args()

    feats_path = os.path.join(args.exp_dir, 'feats.arrow')
    if os.path.exists(feats_path):
        raise SystemExit(f'{feats_path} exists; remove it or use another --exp_dir')

    raw = HFDataset.from_file(os.path.join(args.exp_dir, 'raw.arrow'))
    raw_paths = raw['audio_path']
    languages_all = {language_of(p_) for p_ in raw_paths}
    n_phones, n_speakers, languages = write_vocab_and_speakers(args.exp_dir, raw_paths, languages_all)
    print(f'phones {n_phones}  speakers {n_speakers}  languages {len(languages)}: {languages}')

    selected = select_rows(raw, args.max_utts_per_speaker, args.seed)
    print(f'extracting features for {len(selected)} utterances with {args.workers} workers')
    jobs = [(i, raw_paths[i]) for i in selected]

    os.makedirs(os.path.join(args.exp_dir, 'scalers'), exist_ok=True)
    writer = ArrowWriter(path=feats_path)
    failed = 0
    ctx = mp.get_context('fork')
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for res in tqdm(pool.imap(extract_one, jobs, chunksize=args.chunksize), total=len(jobs), desc='Features'):
            if res is None:
                failed += 1
                continue
            writer.write({'raw_index': res['raw_index'], 'num_frames': res['num_frames'],
                          'f0': res['f0'].tolist(), 'taps': res['taps'].reshape(-1).tolist()})
    writer.finalize()

    stats = compute_speaker_tables(args.exp_dir, raw_paths)
    meta = {'exp_dir': os.path.abspath(args.exp_dir), 'utterances': len(jobs), 'failed': failed,
            'mel': {'sample_rate': SAMPLE_RATE, 'hop_length': 320, 'n_mels': 100},
            'f0_frame_space_ms': F0_FRAME_SPACE_MS, 'speakers_with_stats': len(stats)}
    with open(os.path.join(args.exp_dir, 'extract_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'done: {len(jobs) - failed} utterances written to {feats_path}')


if __name__ == '__main__':
    main()
