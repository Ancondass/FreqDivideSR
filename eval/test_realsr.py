import torch
import os, re, glob, argparse, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from test_set5 import calc_psnr_y, calc_ssim_y, self_ensemble_forward, load_image
from test_benchmark import default_ckpt, load_model

DEFAULT_ROOTS = [
    '/root/shared-nvme/data/Canon/Test',
    '/root/shared-nvme/data/RealSR(V3)/Canon/Test',
    '/root/shared-nvme/data/RealSR(V3)/Canon',
    '/root/shared-nvme/data/RealSR/Canon/Test',
]
SCALES = [2, 3, 4]


def find_pair_root(root):
    out = {}
    for s in SCALES:
        cands = [
            (os.path.join(root, 'LR', f'{s}'), os.path.join(root, 'HR', f'{s}'), False),
            (os.path.join(root, 'LR', f'X{s}'), os.path.join(root, 'HR', f'X{s}'), False),
            (os.path.join(root, f'Test{s}', 'LR'), os.path.join(root, f'Test{s}', 'HR'), False),
            (os.path.join(root, f'{s}'), os.path.join(root, f'{s}'), True),
        ]
        for lr_d, hr_d, mixed in cands:
            if not os.path.isdir(lr_d):
                continue
            if mixed:
                has_hr = glob.glob(os.path.join(lr_d, '*_HR.png'))
                has_lr = glob.glob(os.path.join(lr_d, f'*_LR*.png'))
                if has_hr and has_lr:
                    out[s] = (lr_d, hr_d, mixed)
                    break
            elif lr_d != hr_d and glob.glob(os.path.join(lr_d, '*.png')):
                out[s] = (lr_d, hr_d, mixed)
                break
    return out


def match_hr(lr_name, hr_dir, scale, mixed=False):
    base = os.path.splitext(lr_name)[0]
    if mixed:
        hr_base = re.sub(r'_LR\d*$', '_HR', base)
        for cand in (hr_base, re.sub(r'_LR\d*', '_HR', base)):
            for ext in ('.png', '.jpg'):
                p = os.path.join(hr_dir, cand + ext)
                if os.path.exists(p):
                    return p
        return None
    for cand in (base,
                 base.replace('LR_', 'HR_').replace(f'_x{scale}', ''),
                 base.replace(f'x{scale}', '').replace('LR_', 'HR_')):
        for ext in ('.png', '.jpg'):
            p = os.path.join(hr_dir, cand + ext)
            if os.path.exists(p):
                return p
    return None


def main():
    parser = argparse.ArgumentParser(description='RealSR ')
    parser.add_argument('--data-root', default=None, help='RealSR Canon/Test ')
    parser.add_argument('--ckpt-x2', default=default_ckpt(2))
    parser.add_argument('--ckpt-x3', default=default_ckpt(3))
    parser.add_argument('--ckpt-x4', default=default_ckpt(4))
    parser.add_argument('--border', type=int, default=4)
    parser.add_argument('--self-ensemble', action='store_true')
    parser.add_argument('--save-sr', action='store_true')
    parser.add_argument('--out-root', default='/root/shared-nvme/data/sr_outputs')
    args = parser.parse_args()

    roots = [args.data_root] if args.data_root else DEFAULT_ROOTS
    pairs = {}
    for r in roots:
        if r and os.path.isdir(r):
            pairs = find_pair_root(r)
            if pairs:
                print(f': {r} → x{" x".join(str(s) for s in sorted(pairs))}')
                break
    if not pairs:
        print('  /root/shared-nvme/data/RealSR(V3)/')
        return

    rows = []
    for s in SCALES:
        if s not in pairs:
            print(f'[SKIP] x{s}: ')
            continue
        ckpt_path = getattr(args, f'ckpt_x{s}')
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f'[SKIP] x{s}: checkpoint not found: {ckpt_path}')
            continue
        lr_dir, hr_dir, mixed = pairs[s]
        print(f'\n{"=" * 70}\nx{s}  LR: {lr_dir}  (mixed={mixed})')
        model = load_model(s, ckpt_path)

        if mixed:
            lr_files = sorted(glob.glob(os.path.join(lr_dir, '*_LR*.png'))
                              + glob.glob(os.path.join(lr_dir, '*_LR*.jpg')))
        else:
            lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png'))
                              + glob.glob(os.path.join(lr_dir, '*.jpg')))
        stats = []
        with torch.inference_mode():
            for lr_path in lr_files:
                name = os.path.splitext(os.path.basename(lr_path))[0]
                hp = match_hr(os.path.basename(lr_path), hr_dir, s, mixed=mixed)
                if hp is None:
                    print(f'  [SKIP] no HR pair: {name}')
                    continue
                lr = load_image(lr_path)
                hr = load_image(hp)
                lr_norm = (lr - 0.5) / 0.5
                if args.self_ensemble:
                    sr_norm = self_ensemble_forward(model, lr_norm.to(Config.DEVICE), scale=s)
                else:
                    sr_norm = model(lr_norm.to(Config.DEVICE), scale=s)
                sr = torch.clamp(sr_norm.cpu() * 0.5 + 0.5, 0, 1)

                if args.save_sr:
                    import numpy as np
                    from PIL import Image
                    arr = (sr[0].permute(1, 2, 0).numpy() * 255.0).round().astype('uint8')
                    d = os.path.join(args.out_root, 'SR', 'RealSR', f'X{s}')
                    os.makedirs(d, exist_ok=True)
                    Image.fromarray(arr).save(os.path.join(d, name + '.png'), compress_level=4)

                stats.append((calc_psnr_y(sr, hr, border=args.border),
                              calc_ssim_y(sr, hr, border=args.border)))

        if stats:
            p = sum(x[0] for x in stats) / len(stats)
            ss = sum(x[1] for x in stats) / len(stats)
            rows.append((s, p, ss, len(stats)))
            print(f'  => RealSR x{s}: PSNR {p:.2f} SSIM {ss:.4f} ({len(stats)} imgs, border={args.border})')

    print(f'\n{"=" * 70}\nSummary (Y channel, {args.border}px border, bicubic-trained → real degradation)')
    print(f'{"Scale":<8} {"PSNR":>8} {"SSIM":>8} {"#Imgs":>6}')
    for s, p, ss, n in rows:
        print(f'x{s:<7} {p:8.2f} {ss:8.4f} {n:6d}')


if __name__ == '__main__':
    main()
