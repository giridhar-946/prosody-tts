"""Synthesize speech: espeak-ng text -> phones -> MLMS acoustic model -> Vocos (16 kHz).

Input text file, one utterance per line: '<language><TAB><text>', where <language> is a
name from exp_dir/languages.txt (the accent group the output should follow) and
<text> is phonemized per word with espeak for that language, ASCII words as English
(same conversion as gen_text in f5_tts infer_v5n_multi.py; see phonemize.py).

Usage (repo root):
    python synthesize.py --exp_dir exp --speaker english_female_indtef \
        --text_file sentences.txt --out_dir Synthesized

With --phones_input the text column is already space-separated phone symbols from the
model inventory (exp/phones.txt), which skips espeak.

Pipeline per utterance (same three passes as the earlier version):
    1. duration  -> phone durations (20 ms frames)
    2. F0 + VAD  -> reference F0 contour
    3. log-mel   -> Vocos decoder -> waveform
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from scipy.signal import medfilt

from data import DUR_NORM_MAX, DUR_NORM_MIN, MAX_PHONE_FRAMES, read_name_ids, read_phone_table
from model import MLMSTTS
from utils import arange_list, inv_minmax_norm
from vocoder import DEFAULT_VOCODER_DIR, SAMPLE_RATE, decode, load_vocos

VOICED_THRESHOLD = 50.0    # Hz; frames below this are treated as unvoiced at inference
VAD_THRESHOLD = 0.3
F0_MAX_EMBED = 399


def ids_from_phones(phones, phone_table, sentence_no):
    unknown = sorted({p for p in phones if p not in phone_table})
    if unknown:
        raise SystemExit(f'sentence {sentence_no}: phones not in phones.txt: {unknown}')
    return np.array([phone_table[p] for p in phones], dtype=np.int64)


@torch.no_grad()
def synthesize_log_mel(model, phone_ids, lang_id, spk_id, f0_min, f0_max, device):
    """Returns the predicted log-mel (100, F) on the CPU."""
    L = len(phone_ids)

    def T(a, dtype=torch.long):
        return torch.as_tensor(a, dtype=dtype, device=device).unsqueeze(0)

    phn_t = T(phone_ids)
    mask_t = T(np.ones(L), torch.float32).unsqueeze(-1)
    lang_t = T(np.full(L, lang_id))
    spk_t = T(np.full(L, spk_id))

    # 1. duration
    dur, _, _ = model.forward_duration(phn_t, mask_t, lang_t, spk_t)
    est = inv_minmax_norm(dur[0, :, 0].cpu().numpy(), min_val=DUR_NORM_MIN, max_val=DUR_NORM_MAX)
    phone_frames = np.clip(np.round(est), 1, MAX_PHONE_FRAMES).astype(np.int64)

    repeats = np.repeat(np.arange(L), phone_frames)
    F = len(repeats)
    rep_t = T(np.stack((np.zeros(F), repeats), axis=-1))
    fwd, rev = arange_list(phone_frames)
    fwd_t, rev_t = T(fwd.astype(np.int64)), T(rev.astype(np.int64))
    len_mask_t = T(np.ones(F), torch.float32).unsqueeze(-1)

    # 2. F0 and VAD. The pitch branch does not read the reference F0, so zeros are fine here.
    zero_f0 = torch.zeros(1, F, dtype=torch.long, device=device)
    pred = model(phn_t, mask_t, rep_t, fwd_t, rev_t, len_mask_t, zero_f0, lang_t, spk_t)
    vad = pred['vad'][0, :, 0].cpu().numpy()
    f0 = inv_minmax_norm(pred['pitch'][0, :, 0].cpu().numpy(), min_val=f0_min, max_val=f0_max)
    f0 = medfilt(f0, 3)
    f0 = f0 * (vad > VAD_THRESHOLD)
    f0 = f0.astype(np.int32)
    f0[vad < VAD_THRESHOLD] = 1
    f0[f0 < VOICED_THRESHOLD] = 0
    f0_t = T(np.clip(f0, 0, F0_MAX_EMBED))

    # 3. log-mel
    pred = model(phn_t, mask_t, rep_t, fwd_t, rev_t, len_mask_t, f0_t, lang_t, spk_t)
    return pred['mel'][0].cpu().T.contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', default='exp')
    p.add_argument('--ckpt', default=None, help='default: <exp_dir>/weights/best.pt')
    p.add_argument('--speaker', required=True, help='speaker key, e.g. english_female_indtef (see speakers.txt)')
    p.add_argument('--text_file', required=True)
    p.add_argument('--out_dir', default='Synthesized')
    p.add_argument('--phones_input', action='store_true')
    p.add_argument('--vocoder_dir', default=DEFAULT_VOCODER_DIR)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    exp = args.exp_dir
    ckpt_path = args.ckpt or os.path.join(exp, 'weights', 'best.pt')

    speakers = read_name_ids(os.path.join(exp, 'speakers.txt'))
    languages = read_name_ids(os.path.join(exp, 'languages.txt'))
    if args.speaker not in speakers:
        raise SystemExit(f'unknown speaker {args.speaker}')
    spk_id = speakers[args.speaker]
    with open(os.path.join(exp, 'speaker_stats.json'), encoding='utf-8') as f:
        stats = json.load(f)[args.speaker]

    phone_table = read_phone_table(os.path.join(exp, 'phones.txt'))
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    model = MLMSTTS(cfg['n_phones'], cfg['n_langs'], cfg['n_speakers']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    vocos = load_vocos(args.vocoder_dir, device=device)

    if not args.phones_input:
        # imported here so that --phones_input runs without espeak
        from phonemize import text_to_phones
    with open(args.text_file, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]

    out_dir = os.path.join(args.out_dir, args.speaker)
    os.makedirs(out_dir, exist_ok=True)
    for i, line in enumerate(lines, start=1):
        language, text = line.split('\t', 1)
        if language not in languages:
            raise SystemExit(f'sentence {i}: unknown language {language}; known: {sorted(languages)}')
        if args.phones_input:
            phones = text.split()
        else:
            try:
                phones = text_to_phones(text, set(phone_table), language)
            except ValueError as e:
                raise SystemExit(f'sentence {i}: {e}')
        phone_ids = ids_from_phones(phones, phone_table, i)

        log_mel = synthesize_log_mel(model, phone_ids, languages[language], spk_id,
                                     stats['min_f0'], stats['max_f0'], device)
        wav = decode(vocos, log_mel)
        wav = wav / max(float(wav.abs().max()), 1e-6)

        stem = os.path.join(out_dir, f'{i:03d}_{language}_ep{ckpt["epoch"]:04d}')
        sf.write(stem + '.wav', wav.numpy(), SAMPLE_RATE)
        with open(stem + '.phones.txt', 'w', encoding='utf-8') as f:
            f.write(' '.join(phones) + '\n')
        print(f'[{i}/{len(lines)}] {stem}.wav', flush=True)


if __name__ == '__main__':
    main()
