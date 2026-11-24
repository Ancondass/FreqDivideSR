import torch
import os, glob, time, argparse, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from test_set5 import calc_psnr_y, calc_ssim_y, self_ensemble_forward, load_image
from test_benchmark import default_ckpt, load_model, DATA_ROOT, DATASETS

OUT_ROOT = '/root/shared-nvme/data/sr_outputs'

_lpips_model = None


def get_lpips():
    """ LPIPS (AlexNet). lpips None."""
    global _lpips_model
    if _lpips_model is not None:
        return _lpips_model
    try:
        import lpips as lpips_pkg
    except ImportError:
        return None
    _lpips_model = lpips_pkg.LPIPS(net='alex').to(Config.DEVICE).eval()
    for p in _lpips_model.parameters():
        p.requires_grad = False
    return _lpips_model


def save_png(tensor01, path):
    """tensor [1,3,H,W] ∈ [0,1] → PNG (uint8)."""
    import numpy as np
    from PIL import Image
    arr = (tensor01[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype('uint8')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, compress_level=4)


def main():
    parser = argparse.ArgumentParser(description=' SR + LPIPS/PSNR/SSIM ')
    parser.add_argument('--ckpt-x2', default=default_ckpt(2))
    parser.add_argument('--ckpt-x3', default=default_ckpt(3))
    parser.add_argument('--ckpt-x4', default=default_ckpt(4))
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--scales', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--border', type=int, default=4, help='PSNR/SSIM (LPIPS)')
    parser.add_argument('--self-ensemble', action='store_true')
    parser.add_argument('--no-lpips', action='store_true', help=' SR ')
    parser.add_argument('--save-hr', action='store_true', help=' HR')
    parser.add_argument('--out-root', default=OUT_ROOT)
    args = parser.parse_args()

    lpips_fn = None if args.no_lpips else get_lpips()

    rows = []  # (ds, s, psnr, ssim, lpips, n)
    for s in args.scales:
        ckpt_path = getattr(args, f'ckpt_x{s}')
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f'[SKIP] x{s}: checkpoint not found: {ckpt_path}')
            continue
        print(f'\n{"=" * 70}\nLoading x{s}: {ckpt_path}')
        model = load_model(s, ckpt_path)

        for ds in args.datasets:
            lr_dir = os.path.join(args.data_root, ds, 'LR_bicubic', f'X{s}')
            hr_dir = os.path.join(args.data_root, ds, 'HR_crop', f'X{s}')
            if not os.path.isdir(hr_dir):
                alt = os.path.join(args.data_root, ds, 'HR')
                if os.path.isdir(alt):
                    hr_dir = alt
            if not os.path.isdir(lr_dir):
                print(f'[SKIP] LR dir not found: {lr_dir}')
                continue
            sr_dir = os.path.join(args.out_root, 'SR', ds, f'X{s}')
            hr_out = os.path.join(args.out_root, 'HR', ds, f'X{s}')
            os.makedirs(sr_dir, exist_ok=True)
            if args.save_hr:
                os.makedirs(hr_out, exist_ok=True)

            lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png'))
                              + glob.glob(os.path.join(lr_dir, '*.jpg'))
                              + glob.glob(os.path.join(lr_dir, '*.bmp')))
            stats = []
            with torch.inference_mode():
                for lr_path in lr_files:
                    name = os.path.splitext(os.path.basename(lr_path))[0]
                    hr_path = os.path.join(hr_dir, os.path.basename(lr_path))
                    if not os.path.exists(hr_path):
                        alt = os.path.splitext(hr_path)[0] + '.jpg'
                        if not os.path.exists(alt):
                            print(f'  [SKIP] no HR: {name}')
                            continue
                        hr_path = alt

                    lr = load_image(lr_path)
                    hr = load_image(hr_path)
                    lr_norm = (lr - 0.5) / 0.5
                    if args.self_ensemble:
                        sr_norm = self_ensemble_forward(model, lr_norm.to(Config.DEVICE), scale=s)
                    else:
                        sr_norm = model(lr_norm.to(Config.DEVICE), scale=s)
                    sr = torch.clamp(sr_norm.cpu() * 0.5 + 0.5, 0, 1)

                    save_png(sr, os.path.join(sr_dir, name + '.png'))
                    if args.save_hr:
                        save_png(hr, os.path.join(hr_out, name + '.png'))

                    psnr = calc_psnr_y(sr, hr, border=args.border)
                    ssim = calc_ssim_y(sr, hr, border=args.border)
                    lp = float('nan')
                    if lpips_fn is not None:
                        lv = lpips_fn(sr.to(Config.DEVICE) * 2 - 1, hr.to(Config.DEVICE) * 2 - 1)
                        lp = lv.item()
                    stats.append((psnr, ssim, lp))

            if not stats:
                continue
            psnr_m = sum(x[0] for x in stats) / len(stats)
            ssim_m = sum(x[1] for x in stats) / len(stats)
            lps = [x[2] for x in stats if x[2] == x[2]]
            lp_m = sum(lps) / len(lps) if lps else float('nan')
            rows.append((ds, s, psnr_m, ssim_m, lp_m, len(stats)))
            lp_s = f'{lp_m:.4f}' if lps else '  n/a'
            print(f'  {ds} x{s}: PSNR {psnr_m:.2f} SSIM {ssim_m:.4f} LPIPS {lp_s} ({len(stats)} imgs)')

    print(f'\n{"=" * 70}')
    print(f'{"Dataset":<10} {"Scale":<6} {"PSNR":>8} {"SSIM":>8} {"LPIPS↓":>8} {"#Imgs":>6}')
    print('-' * 50)
    for ds, s, p, ss, lp, n in rows:
        lp_s = f'{lp:.4f}' if lp == lp else 'n/a'
        print(f'{ds:<10} x{s:<5} {p:8.2f} {ss:8.4f} {lp_s:>8} {n:6d}')

    md = os.path.join(args.out_root, 'lpips_summary.md')
    os.makedirs(args.out_root, exist_ok=True)
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# SR outputs + LPIPS summary\n\n')
        f.write(f'self_ensemble={args.self_ensemble}, lpips=alex, rgb, no border\n\n')
        f.write('| Dataset | Scale | PSNR | SSIM | LPIPS | #Imgs |\n|---|---|---|---|---|---|\n')
        for ds, s, p, ss, lp, n in rows:
            lp_s = f'{lp:.4f}' if lp == lp else ''
            f.write(f'| {ds} | x{s} | {p:.2f} | {ss:.4f} | {lp_s} | {n} |\n')
    print(f'\n: {md}')
    print(f'SR: {args.out_root}/SR/')


if __name__ == '__main__':
    main()
