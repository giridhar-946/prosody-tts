"""Train the MLMS TTS acoustic model on the F5-TTS arrow data.

Usage (after extract_features.py has written <exp_dir>/feats.arrow):
    python train.py --arrow_dir /path/to/arrow_dir --exp_dir exp

Batches are built as in F5 DynamicBatchSampler: a budget of mel frames per batch
(--frames_per_batch, 50 frames = 1 s), capped at --max_samples utterances.

Checkpoints go to <exp_dir>/weights/:
    weights-NNNN.pt   model weights and config, every epoch (inference only)
    best.pt           lowest validation loss
    last.pt           full state (optimizer, scheduler, counters) for --resume

After each validation pass one random dev utterance is synthesized with the current model.
<exp_dir>/samples/ gets ep{NNNN}_<speaker>_<utt>.wav (synthesized) for every epoch, and
original_<speaker>_<utt>.wav (the source wav, copied once).
"""
import argparse
import json
import os
import random
import shutil
import time

import soundfile as sf
import torch
from torch.utils.data import DataLoader

from data import (DynamicBatchSampler, MLMSDataset, collate_fn, load_feats, load_speaker_tables,
                  read_name_ids, read_phone_table, speaker_of)
from model import MLMSTTS
from synthesize import synthesize_log_mel
from vocoder import SAMPLE_RATE, decode, load_vocos

LOSS_WEIGHTS = {'dur': 20.0, 'pitch': 10.0, 'vad': 1.0, 'mel': 10.0, 'taps': 1.0}


def compute_loss(model, inputs, targets):
    pred = model(**inputs)
    F_ = torch.nn.functional
    losses = {
        'dur': F_.l1_loss(pred['dur'], targets['dur']),
        'pitch': F_.l1_loss(pred['pitch'], targets['pitch']),
        'vad': F_.binary_cross_entropy(pred['vad'].clamp(1e-7, 1 - 1e-7), targets['vad']),
        'mel': F_.l1_loss(pred['mel'], targets['mel']),
        'taps': F_.mse_loss(pred['taps'], targets['taps']),
    }
    total = sum(LOSS_WEIGHTS[k] * v for k, v in losses.items()) + model.l2_penalty()
    return total, {k: v.item() for k, v in losses.items()}


def run_epoch(model, loader, device, optimizer=None, log_every=0):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for step, (inputs, targets) in enumerate(loader, start=1):
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            loss, parts = compute_loss(model, inputs, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
            if training and log_every and step % log_every == 0:
                print(f'  step {step}  loss {loss.item():.4f}  ' +
                      '  '.join(f'{k} {v:.3f}' for k, v in parts.items()), flush=True)
    return total / max(n, 1)


def save_validation_sample(epoch, model, dev_ds, sample_index, stats, vocos, samples_dir, device):
    """Synthesizes dev utterance `sample_index` and writes it next to its original."""
    item = dev_ds[sample_index]
    raw_row = dev_ds.raw[dev_ds.items[sample_index][0]]
    path = raw_row['audio_path']
    spk = speaker_of(path)
    stem = f'{spk}_{os.path.splitext(os.path.basename(path))[0]}'

    original = os.path.join(samples_dir, f'original_{stem}.wav')
    if not os.path.exists(original):
        shutil.copy(path, original)

    model.eval()
    log_mel = synthesize_log_mel(model, item['phones'], item['lang_id'], item['spk_id'],
                                 stats[spk]['min_f0'], stats[spk]['max_f0'], device)
    wav = decode(vocos, log_mel)
    wav = wav / max(float(wav.abs().max()), 1e-6)
    sf.write(os.path.join(samples_dir, f'ep{epoch:04d}_{stem}.wav'), wav.numpy(), SAMPLE_RATE)
    print(f'  validation sample epoch {epoch}: {stem}', flush=True)


def split_indices(raw_indices, dev_fraction, max_dev, seed):
    rng = random.Random(seed)
    shuffled = list(raw_indices)
    rng.shuffle(shuffled)
    n_dev = min(max_dev, max(1, int(len(shuffled) * dev_fraction)))
    return shuffled[n_dev:], shuffled[:n_dev]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', default='exp')
    p.add_argument('--arrow_dir', required=True, help='folder with raw.arrow (see docs/data.md)')
    p.add_argument('--out_dir', default=None, help='default: <exp_dir>/weights')
    p.add_argument('--frames_per_batch', type=int, default=2500,
                   help='mel frames (20 ms each) per batch; 2500 = 50 s of audio')
    p.add_argument('--max_samples', type=int, default=32)
    p.add_argument('--dev_fraction', type=float, default=0.005)
    p.add_argument('--max_dev', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=10, help='early stopping patience (epochs)')
    p.add_argument('--lr_patience', type=int, default=30, help='ReduceLROnPlateau patience')
    p.add_argument('--lr_factor', type=float, default=0.1)
    p.add_argument('--min_lr', type=float, default=1e-8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--resume', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    exp = args.exp_dir
    out_dir = args.out_dir or os.path.join(exp, 'weights')
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    phone_table = read_phone_table(os.path.join(exp, 'phones.txt'))
    spk2id = read_name_ids(os.path.join(exp, 'speakers.txt'))
    lang2id = read_name_ids(os.path.join(exp, 'languages.txt'))
    stats, scalers = load_speaker_tables(exp)
    feats = load_feats(exp)

    train_raw, dev_raw = split_indices([int(r) for r in feats['raw_index']],
                                       args.dev_fraction, args.max_dev, args.seed)
    train_ds = MLMSDataset(args.arrow_dir, feats, train_raw, phone_table, spk2id, lang2id, stats, scalers)
    dev_ds = MLMSDataset(args.arrow_dir, feats, dev_raw, phone_table, spk2id, lang2id, stats, scalers)
    print(f'train utts {len(train_ds)} (dropped {train_ds.dropped})  dev utts {len(dev_ds)}')

    train_sampler = DynamicBatchSampler(train_ds, args.frames_per_batch, args.max_samples, args.seed, shuffle=True)
    dev_sampler = DynamicBatchSampler(dev_ds, args.frames_per_batch, args.max_samples, args.seed, shuffle=False)
    print(f'batches: train {len(train_sampler)}  dev {len(dev_sampler)}')
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    dev_loader = DataLoader(dev_ds, batch_sampler=dev_sampler, collate_fn=collate_fn,
                            num_workers=args.workers, pin_memory=True)

    config = {'n_phones': len(phone_table), 'n_langs': len(lang2id), 'n_speakers': len(spk2id)}
    model = MLMSTTS(**config).to(device)

    samples_dir = os.path.join(exp, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    vocos = load_vocos(device=device)
    sample_rng = random.Random(args.seed)
    print(f'validation samples: one random dev utterance per epoch -> {samples_dir}')
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr, threshold=0.0, threshold_mode='abs')

    start_epoch, best_val, wait = 1, float('inf'), 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch, best_val, wait = ckpt['epoch'] + 1, ckpt['best_val'], ckpt['wait']
        print(f'resumed from {args.resume} at epoch {start_epoch}')

    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({**config, 'args': vars(args), 'dropped_train': train_ds.dropped,
                   'dropped_dev': dev_ds.dropped}, f, indent=2)
    print(f'model {config}  trainable params {sum(p_.numel() for p_ in trainable):,}')

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer, log_every=args.log_every)
        val_loss = run_epoch(model, dev_loader, device)
        scheduler.step(val_loss)
        sample_index = sample_rng.randrange(len(dev_ds))
        save_validation_sample(epoch, model, dev_ds, sample_index, stats, vocos, samples_dir, device)

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'wait': wait,
                 'best_val': best_val, 'config': config}
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            state['best_val'] = best_val
            torch.save(state, os.path.join(out_dir, 'best.pt'))
        else:
            wait += 1
        torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch},
                   os.path.join(out_dir, f'weights-{epoch:04d}.pt'))
        torch.save(state, os.path.join(out_dir, 'last.pt'))

        print(f'epoch {epoch:04d}  train {train_loss:.4f}  val {val_loss:.4f}  '
              f'lr {optimizer.param_groups[0]["lr"]:.1e}  best {best_val:.4f}  {time.time() - t0:.0f}s',
              flush=True)
        if wait >= args.patience:
            print(f'early stopping at epoch {epoch}')
            break


if __name__ == '__main__':
    main()
