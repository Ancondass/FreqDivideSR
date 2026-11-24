
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from measure_complexity import (default_ckpt, PROFILES, count_flops, lr_size,
                                load_model, pick_flops)
from measure_complexity import measure_latency as latency_stats
from test_benchmark import DATA_ROOT, test_dataset

SEP = '=' * 92


def resolve_dirs(data_root, ds, scale):
    """ test_benchmark.py """
    lr_dir = os.path.join(data_root, ds, 'LR_bicubic', f'X{scale}')
    hr_dir = os.path.join(data_root, ds, 'HR_crop', f'X{scale}')
    if not os.path.isdir(hr_dir):
        alt = os.path.join(data_root, ds, 'HR')
        if os.path.isdir(alt):
            print(f' [Fallback] {ds}/HR_crop/X{scale} → {alt}')
            hr_dir = alt
    return lr_dir, hr_dir


def eval_datasets(model, scale, datasets, data_root, border):
    """ {dataset: (psnr, ssim, n, seconds)}"""
    out = {}
    for ds in datasets:
        lr_dir, hr_dir = resolve_dirs(data_root, ds, scale)
        if not os.path.isdir(lr_dir):
            print(f' [SKIP] {ds}: LR {lr_dir}')
            continue
        res = test_dataset(model, scale, lr_dir, hr_dir, border,
                           self_ensemble=False, detail=False)
        if res is None:
            print(f' [SKIP] {ds}: ')
            continue
        out[ds] = res
    return out


def measure_latency(model, x, device, warmup, iters):
    mean, _std, _median = latency_stats(model, x, device, warmup, iters,
                                        self_ensemble=False, scale=None)
    return mean


def linear_fit(xs, ys):
    """ y = a + b·x, (a, b)"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser(description='ODE / ( §4.6 5)')
    ap.add_argument('--scale', type=int, default=4, choices=[2, 3, 4])
    ap.add_argument('--ckpt', default=None, help=' ()')
    ap.add_argument('--device', default=None, help='cuda / cuda:0 / cpu')
    ap.add_argument('--steps', nargs='+', type=int, default=[24, 12, 8, 6, 4, 2, 1],
                    help=' ( 24)')
    ap.add_argument('--datasets', nargs='+', default=['Set5', 'Urban100'])
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--border', type=int, default=4)
    ap.add_argument('--profiles', nargs='+', default=['48', '720p'], choices=list(PROFILES),
                    help='MACs ')
    ap.add_argument('--lat-profiles', nargs='+', default=['48'], choices=list(PROFILES),
                    help=' (720p 1,)')
    ap.add_argument('--no-flops', action='store_true', help=' MACs ( PSNR)')
    ap.add_argument('--no-eval', action='store_true', help=' PSNR ()')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else Config.DEVICE
    if device.type == 'cpu':
        args.iters = min(args.iters, 10)

    ckpt_path = args.ckpt or default_ckpt(args.scale)
    print(SEP)
    print(f': {device}')
    print(f': x{args.scale}')
    print(f': {ckpt_path}')
    print(f': {" ".join(str(s) for s in args.steps)}')
    print(f'MACs: {" ".join(args.profiles) if not args.no_flops else ""}')
    print(f': {" ".join(args.lat_profiles)} '
          f'(warmup {args.warmup} / iters {args.iters})')
    if not args.no_eval:
        print(f': {args.data_root}')
        print(f': {" ".join(args.datasets)}')
    print(SEP)

    if not os.path.exists(ckpt_path):
        print(f'[Error]: {ckpt_path}')
        return 1

    model, ablation = load_model(args.scale, ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f': {n_params:,} ({n_params / 1e6:.2f} M) '
          f'ablation={sorted(ablation) if ablation else " ()"}')

    if not hasattr(model, 'ode_block') or not hasattr(model.ode_block, 'num_steps'):
        return 1

    method = getattr(model.ode_block, 'method', 'rk2')
    evals_per_step = 2 if method == 'rk2' else 4
    train_steps = model.ode_block.num_steps
    print(f'ODE: method={method}, N={train_steps}, '
          f'NFE = {evals_per_step}×N = {train_steps * evals_per_step} f_θ ')

    need = set(args.lat_profiles)
    if not args.no_flops:
        need |= set(args.profiles)
    inputs, sizes = {}, {}
    for key in need:
        hr_h, hr_w, _ = PROFILES[key]
        lh, lw, _exact = lr_size(hr_h, hr_w, args.scale)
        sizes[key] = (lh, lw)
        inputs[key] = torch.randn(1, 3, lh, lw, device=device)

    rows = []
    for n in sorted(args.steps, reverse=True):
        model.ode_block.num_steps = n
        rec = {'steps': n, 'nfe': n * evals_per_step, 'macs': {}, 'lat': {}, 'eval': {}}

        if not args.no_flops:
            for key in args.profiles:
                f, _src = pick_flops(count_flops(model, inputs[key]))
                rec['macs'][key] = f
        for key in args.lat_profiles:
            rec['lat'][key] = measure_latency(model, inputs[key], device,
                                              args.warmup, args.iters)
        if not args.no_eval:
            rec['eval'] = eval_datasets(model, args.scale, args.datasets,
                                        args.data_root, args.border)

        msg = f'  N={n:<3} NFE={rec["nfe"]:<4}'
        for key in args.profiles:
            f = rec['macs'].get(key)
            msg += f'  {key}: {f:7.1f} G' if f is not None else f' {key}: '
        for key in args.lat_profiles:
            msg += f' [{key}] {rec["lat"][key]:8.2f} ms'
        for ds, (p, s, _cnt, _t) in rec['eval'].items():
            msg += f'  {ds} {p:6.2f}/{s:.4f}'
        print(msg, flush=True)
        rows.append(rec)

    model.ode_block.num_steps = train_steps

    cols = ['N', 'NFE']
    for key in args.profiles:
        if rows[0]['macs'].get(key) is not None:
            cols.append(f'MACs@{key}')
    for key in args.lat_profiles:
        cols.append(f'@{key}')
    for ds in args.datasets:
        if not args.no_eval:
            cols += [f'{ds} PSNR', f'{ds} SSIM']

    print(f'\n{SEP}\n ( {args.border}px, Y,)')
    widths = [4, 6] + [13] * (len(cols) - 2)
    print(''.join(f'{c:>{w}}' for c, w in zip(cols, widths)))
    print('-' * sum(widths))
    base = {}
    if not args.no_eval:
        for ds in args.datasets:
            for rec in rows:
                if ds in rec['eval']:
                    base[ds] = rec['eval'][ds][0]
                    break
    for rec in rows:
        cells = [str(rec['steps']), str(rec['nfe'])]
        for key in args.profiles:
            f = rec['macs'].get(key)
            if rows[0]['macs'].get(key) is not None:
                cells.append(f'{f:.1f}' if f is not None else '')
        for key in args.lat_profiles:
            cells.append(f'{rec["lat"][key]:.2f}')
        if not args.no_eval:
            for ds in args.datasets:
                if ds in rec['eval']:
                    p = rec['eval'][ds][0]
                    cells += [f'{p:.2f}', f'{p - base[ds]:+.2f}']
                else:
                    cells += ['', '']
        print(''.join(f'{c:>{w}}' for c, w in zip(cells, widths)))

    key0 = next((k for k in args.profiles if rows[0]['macs'].get(k) is not None), None)
    if len(rows) >= 2 and key0:
        xs = [r['steps'] for r in rows]
        ys = [r['macs'][key0] for r in rows]
        a, b = linear_fit(xs, ys)
        node = b * train_steps
        print(f'\n{SEP}\n ( {key0}, MACs ≈ A + B·N):')
        print(f' NODE A = {a:.1f} G ()')
        print(f' NODE B = {b:.3f} G/ → N={train_steps} {node:.1f} G')
        if a + node > 0:
            print(f' N={train_steps} NODE {node / (a + node) * 100:.1f}%')
        print(f' N: {n_params / 1e6:.2f} M ()')

    # ---------------- Markdown ----------------
    print(f'\n{SEP}\nMarkdown:')
    md = ['N', 'NFE'] + [f'MACs@{k}(G)' for k in args.profiles if rows[0]['macs'].get(k)] \
         + [f'@{k}(ms)' for k in args.lat_profiles]
    if not args.no_eval:
        for ds in args.datasets:
            md += [f'{ds} PSNR', f'{ds} SSIM']
    print('| ' + ' | '.join(md) + ' |')
    print('|' + '---|' * len(md))
    for rec in rows:
        cells = [str(rec['steps']), str(rec['nfe'])]
        for key in args.profiles:
            f = rec['macs'].get(key)
            if rows[0]['macs'].get(key) is not None:
                cells.append(f'{f:.1f}' if f is not None else '')
        for key in args.lat_profiles:
            cells.append(f'{rec["lat"][key]:.2f}')
        if not args.no_eval:
            for ds in args.datasets:
                if ds in rec['eval']:
                    p, s, _c, _t = rec['eval'][ds]
                    cells += [f'{p:.2f}', f'{s:.4f}']
                else:
                    cells += ['', '']
        print('| ' + ' | '.join(cells) + ' |')

    print(f'\n[] PSNR N={train_steps}, '
          f', §4.6 5')
    print(SEP)
    return 0


if __name__ == '__main__':
    sys.exit(main())
