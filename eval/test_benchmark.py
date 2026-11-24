import torch
import os, glob, time, argparse, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from sr_model import create_model
from test_set5 import calc_psnr_y, calc_ssim_y, self_ensemble_forward, load_image, discover_ckpt

DATA_ROOT = '/root/shared-nvme/data/data'
DATASETS = ['Set5', 'Set14', 'BSD100', 'Urban100']

CKPT_ROOT = Config.CKPT_ROOT
CKPT_OVERRIDE = {
    2: f'{CKPT_ROOT}/logs/x2/best_ep300_x2_psnr34.77_ssim0.9339.pth',
    3: f'{CKPT_ROOT}/logs/x3/best_ep300_x3_psnr30.90_ssim0.8631.pth',
    4: f'{CKPT_ROOT}/logs_abl/full/x4/best_ep300_x4_psnr30.52_ssim0.8410.pth',
}


def default_ckpt(scale):
    """ checkpoint: CKPT_OVERRIDE,."""
    p = CKPT_OVERRIDE.get(scale)
    if p and os.path.exists(p):
        return p
    if p:
        print(f'[Warn] x{scale}: {p} → ')
    return discover_ckpt(scale)


def load_model(scale, ckpt_path):
    """ test_set5.py (, key)."""
    Config.SCALES = [scale]
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
    ablation = ckpt.get('ablation') if isinstance(ckpt, dict) else None
    model = create_model(scales=[scale], ablation=ablation)
    state = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    if any(k.startswith('yolo11_backbone.') for k in state.keys()):
        state = {k.replace('yolo11_backbone.', 'global_branch.'): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  [Warn] Missing keys ({len(missing)}): {missing[:3]}... (random init OK)')
    if unexpected:
        print(f'  [Warn] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')
    model.eval()
    return model


def test_dataset(model, scale, lr_dir, hr_dir, border, self_ensemble, detail):
    """ (avg_psnr, avg_ssim, n, total_time)."""
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.png'))
                      + glob.glob(os.path.join(lr_dir, '*.jpg'))
                      + glob.glob(os.path.join(lr_dir, '*.bmp')))
    results = []
    total_time = 0.0
    with torch.inference_mode():
        for lr_path in lr_files:
            name = os.path.splitext(os.path.basename(lr_path))[0]
            hr_path = os.path.join(hr_dir, os.path.basename(lr_path))
            if not os.path.exists(hr_path):
                alt = os.path.splitext(hr_path)[0] + '.jpg'
                hr_path = alt if os.path.exists(alt) else None
            if hr_path is None or not os.path.exists(hr_path):
                print(f'    [SKIP] HR not found for {name}')
                continue

            lr = load_image(lr_path)
            hr = load_image(hr_path)
            lr_norm = (lr - 0.5) / 0.5
            t0 = time.time()
            if self_ensemble:
                sr_norm = self_ensemble_forward(model, lr_norm.to(Config.DEVICE), scale=scale)
            else:
                sr_norm = model(lr_norm.to(Config.DEVICE), scale=scale)
            torch.cuda.synchronize()
            elapsed = time.time() - t0

            sr = torch.clamp(sr_norm.cpu() * 0.5 + 0.5, 0, 1)
            psnr = calc_psnr_y(sr, hr, border=border)
            ssim = calc_ssim_y(sr, hr, window_size=11, border=border)
            results.append((name, psnr, ssim))
            total_time += elapsed
            if detail:
                print(f'    {name:<24} {psnr:8.2f} {ssim:8.4f} {elapsed:7.3f}s')

    if not results:
        return None
    avg_psnr = sum(r[1] for r in results) / len(results)
    avg_ssim = sum(r[2] for r in results) / len(results)
    return avg_psnr, avg_ssim, len(results), total_time


def main():
    parser = argparse.ArgumentParser(description=' × ')
    parser.add_argument('--ckpt-x2', default=default_ckpt(2), help='x2 ')
    parser.add_argument('--ckpt-x3', default=default_ckpt(3), help='x3 ')
    parser.add_argument('--ckpt-x4', default=default_ckpt(4), help='x4 ')
    parser.add_argument('--data-root', default=DATA_ROOT, help='')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help=f' ( {" ".join(DATASETS)})')
    parser.add_argument('--scales', nargs='+', type=int, default=[2, 3, 4],
                        help=' ( 2 3 4)')
    parser.add_argument('--border', type=int, default=4, help='PSNR/SSIM ')
    parser.add_argument('--self-ensemble', action='store_true', help='8 Self-Ensemble')
    parser.add_argument('--detail', action='store_true', help='')
    parser.add_argument('--latency', action='store_true',
                        help=' (ms/) Table 4 ')
    args = parser.parse_args()

    ckpts = {s: getattr(args, f'ckpt_x{s}') for s in args.scales}
    srcs = {s: ('override' if ckpts[s] == CKPT_OVERRIDE.get(s) else 'CLI/auto')
            for s in args.scales}
    print(f'Device:        {Config.DEVICE}')
    print(f'Self-Ensemble: {"ON" if args.self_ensemble else "OFF"}')
    print(f'Data root:     {args.data_root}')
    print(f'Datasets:      {" ".join(args.datasets)}')
    print(f'Scales:        {" ".join(f"x{s}" for s in args.scales)}')
    print(f'Checkpoints:   ' + '  '.join(f'x{s}={srcs[s]}' for s in args.scales))

    rows = []  # (dataset, scale, psnr, ssim, n, time)
    for s in args.scales:
        ckpt_path = ckpts.get(s)
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f'\n[SKIP] x{s}: checkpoint not found: {ckpt_path}')
            continue
        print(f'\n{"=" * 70}')
        print(f'Loading x{s} checkpoint: {ckpt_path}')
        model = load_model(s, ckpt_path)
        print(f'  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

        for ds in args.datasets:
            lr_dir = os.path.join(args.data_root, ds, 'LR_bicubic', f'X{s}')
            hr_dir = os.path.join(args.data_root, ds, 'HR_crop', f'X{s}')
            if not os.path.isdir(hr_dir):
                alt = os.path.join(args.data_root, ds, 'HR')
                if os.path.isdir(alt):
                    print(f'  [Fallback] {ds}/HR_crop/X{s} missing → {alt}')
                    hr_dir = alt
            print(f'\n  --- {ds} x{s} ---')
            print(f'  LR: {lr_dir} | HR: {hr_dir}')
            if not os.path.isdir(lr_dir):
                print(f'  [SKIP] LR dir not found: {lr_dir}')
                continue
            res = test_dataset(model, s, lr_dir, hr_dir, args.border,
                               args.self_ensemble, args.detail)
            if res is None:
                print(f'  [SKIP] no valid image pairs')
                continue
            psnr, ssim, n, t = res
            rows.append((ds, s, psnr, ssim, n, t))
            latency_ms = t / n * 1000 if n else 0
            print(f'  => PSNR {psnr:.2f} dB | SSIM {ssim:.4f} | {n} imgs | {t:.1f}s'
                  + (f' | {latency_ms:.1f} ms/img' if args.latency else ''))

    print(f'\n{"=" * 70}')
    print('Summary (Y channel, %dpx border)' % args.border)
    if args.latency:
        print(f'{"Dataset":<10} {"Scale":<6} {"PSNR":>8} {"SSIM":>8} {"#Imgs":>6} {"Lat(ms)":>9}')
    else:
        print(f'{"Dataset":<10} {"Scale":<6} {"PSNR":>8} {"SSIM":>8} {"#Imgs":>6} {"Time":>8}')
    print('-' * 70)
    for ds, s, psnr, ssim, n, t in rows:
        if args.latency:
            print(f'{ds:<10} x{s:<5} {psnr:8.2f} {ssim:8.4f} {n:6d} {t / n * 1000:9.1f}')
        else:
            print(f'{ds:<10} x{s:<5} {psnr:8.2f} {ssim:8.4f} {n:6d} {t:7.1f}s')
    print('-' * 70)

    print('\nMarkdown table:')
    print('| Dataset | Scale | PSNR | SSIM | #Imgs |')
    for ds, s, psnr, ssim, n, t in rows:
        print(f'| {ds} | x{s} | {psnr:.2f} | {ssim:.4f} | {n} |')


if __name__ == '__main__':
    main()
