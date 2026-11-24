import torch
import torch.nn.functional as F
import os, glob, math, time, argparse, re
from PIL import Image
import numpy as np

from config import Config
from sr_model import create_model


MATLAB_R, MATLAB_G, MATLAB_B = 65.481, 128.553, 24.966
MATLAB_Y_OFFSET = 16.0


def rgb_to_ycbcr(tensor):
    """MATLAB rgb2ycbcr Y: BT.601 + 16, tensor ∈ [0,1]"""
    y = (MATLAB_Y_OFFSET / 255.0 +
         (MATLAB_R * tensor[:, 0:1, :, :] +
          MATLAB_G * tensor[:, 1:2, :, :] +
          MATLAB_B * tensor[:, 2:3, :, :]) / 255.0)
    return y


def calc_psnr_y(sr, hr, border=4):
    sr_y = rgb_to_ycbcr(sr)[:, :, border:-border, border:-border]
    hr_y = rgb_to_ycbcr(hr)[:, :, border:-border, border:-border]
    mse = torch.mean((sr_y - hr_y) ** 2)
    if mse == 0:
        return 100.0
    return 10 * math.log10(1 / mse.item())


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                           for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D.expand(channel, 1, window_size, window_size).contiguous()


def calc_ssim_y(sr, hr, window_size=11, border=4):
    sr_y = rgb_to_ycbcr(sr)[:, :, border:-border, border:-border]
    hr_y = rgb_to_ycbcr(hr)[:, :, border:-border, border:-border]
    window = create_window(window_size, 1).to(sr_y.device).type_as(sr_y)
    mu1 = F.conv2d(sr_y, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(hr_y, window, padding=window_size // 2, groups=1)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(sr_y * sr_y, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(hr_y * hr_y, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(sr_y * hr_y, window, padding=window_size // 2, groups=1) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def _forward_batch(model, xs, scale):
    """ batch (OOM,)."""
    if len(xs) == 1:
        return [model(xs[0], scale=scale)]
    try:
        out = model(torch.cat(xs, dim=0), scale=scale)
        return list(out.chunk(len(xs), dim=0))
    except RuntimeError as err:
        if 'out of memory' not in str(err).lower():
            raise
        torch.cuda.empty_cache()
        return [model(t, scale=scale) for t in xs]


def self_ensemble_forward(model, x, scale):
    x_t = x.transpose(2, 3)
    groups = [
        ([x,
          torch.flip(x, [3]),
          torch.flip(x, [2]),
          torch.flip(torch.flip(x, [3]), [2])],
         lambda ys: [ys[0],
                     torch.flip(ys[1], [3]),
                     torch.flip(ys[2], [2]),
                     torch.flip(torch.flip(ys[3], [3]), [2])]),
        ([x_t,
          torch.flip(x_t, [3]),
          torch.flip(x_t, [2]),
          torch.flip(torch.flip(x_t, [3]), [2])],
         lambda ys: [ys[0].transpose(2, 3),
                     torch.flip(ys[1], [3]).transpose(2, 3),
                     torch.flip(ys[2], [2]).transpose(2, 3),
                     torch.flip(torch.flip(ys[3], [3]), [2]).transpose(2, 3)]),
    ]
    outs = []
    for xs, inverse in groups:
        outs += inverse(_forward_batch(model, xs, scale))
    return torch.stack(outs).mean(0)


def load_image(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def discover_ckpt(scale, root=None):
    root = root or getattr(Config, 'CKPT_ROOT', None)
    if not root:
        return None
    patterns = [
        os.path.join(root, 'logs', f'x{scale}', 'best_ep*.pth'),
        os.path.join(root, 'logs', f'x{scale}', '*.pth'),
        os.path.join(root, f'x{scale}', '*.pth'),
        os.path.join(root, 'logs', f'best_*x{scale}*.pth'),
        os.path.join(root, f'*x{scale}*.pth'),
    ]
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            def _key(f):
                base = os.path.basename(f)
                m = re.search(r'psnr([-\d.]+)', base)
                p = float(m.group(1)) if m else -1.0
                e = re.search(r'ep(\d+)', base)
                return (p, int(e.group(1)) if e else 0)
            return max(files, key=_key)
    return None


def main():
    parser = argparse.ArgumentParser(description='Set5 ')
    parser.add_argument('checkpoint', help=' (.pth)')
    parser.add_argument('--lr_dir', default=None, help='LR ( Set5/LR_bicubic/X{scale})')
    parser.add_argument('--hr_dir', default=None, help='HR ( Set5/HR_crop/X{scale}, Set5/HR)')
    parser.add_argument('--scale', type=int, default=4, help='')
    parser.add_argument('--border', type=int, default=4, help='PSNR/SSIM ')
    parser.add_argument('--self-ensemble', action='store_true', help='8 Self-Ensemble (+0.1~0.2 dB)')
    args = parser.parse_args()

    if args.lr_dir is None:
        args.lr_dir = f'/root/shared-nvme/data/data/Set5/LR_bicubic/X{args.scale}'
    if args.hr_dir is None:
        args.hr_dir = f'/root/shared-nvme/data/data/Set5/HR_crop/X{args.scale}'
        if not os.path.isdir(args.hr_dir):
            args.hr_dir = '/root/shared-nvme/data/data/Set5/HR'

    device = Config.DEVICE
    print(f'Device:        {device}')
    print(f'Checkpoint:    {args.checkpoint}')
    print(f'LR dir:        {args.lr_dir}')
    print(f'HR dir:        {args.hr_dir}')
    print(f'Scale:         x{args.scale}')
    print(f'Self-Ensemble: {"ON" if args.self_ensemble else "OFF"}')

    if args.scale not in Config.SCALES:
        print(f'Error: --scale {args.scale} not in Config.SCALES {Config.SCALES}')
        return
    Config.SCALES = [args.scale]
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_ablation = ckpt.get('ablation') if isinstance(ckpt, dict) else None
    model = create_model(scales=[args.scale], ablation=ckpt_ablation)
    state = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    if any(k.startswith('yolo11_backbone.') for k in state.keys()):
        state = {k.replace('yolo11_backbone.', 'global_branch.'): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[Warn] Missing keys ({len(missing)}): {missing[:3]}... (random init OK)')
    if unexpected:
        print(f'[Warn] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')
    model.eval()
    print(f'Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    print(f'\n[Check] LR dir exists: {os.path.isdir(args.lr_dir)} | HR dir exists: {os.path.isdir(args.hr_dir)}')
    lr_files = sorted(glob.glob(os.path.join(args.lr_dir, '*.png'))
                      + glob.glob(os.path.join(args.lr_dir, '*.jpg'))
                      + glob.glob(os.path.join(args.lr_dir, '*.bmp')))
    print(f'Found {len(lr_files)} LR images')
    if len(lr_files) == 0:
        return

    results = []
    total_time = 0.0

    print(f'\n{"="*60}')
    print(f'{"Image":<12} {"LR Size":<14} {"PSNR":>8} {"SSIM":>8} {"Time":>8}')
    print(f'{"-"*60}')

    with torch.inference_mode():
        for lr_path in lr_files:
            name = os.path.splitext(os.path.basename(lr_path))[0]

            hr_path = os.path.join(args.hr_dir, os.path.basename(lr_path))
            if not os.path.exists(hr_path):
                alt = os.path.splitext(hr_path)[0] + '.jpg'
                hr_path = alt if os.path.exists(alt) else None
            if hr_path is None or not os.path.exists(hr_path):
                print(f'  [SKIP] HR not found for {name}')
                continue

            lr = load_image(lr_path)
            hr = load_image(hr_path)
            _, _, h, w = lr.shape
            lr_size = f'{w}x{h}'

            lr_norm = (lr - 0.5) / 0.5
            t0 = time.time()
            if args.self_ensemble:
                sr_norm = self_ensemble_forward(model, lr_norm.to(device), scale=args.scale)
            else:
                sr_norm = model(lr_norm.to(device), scale=args.scale)
            torch.cuda.synchronize()
            elapsed = time.time() - t0

            sr = sr_norm.cpu() * 0.5 + 0.5  # [-1,1] → [0,1]
            sr = torch.clamp(sr, 0, 1)
            psnr = calc_psnr_y(sr, hr, border=args.border)
            ssim = calc_ssim_y(sr, hr, border=args.border)

            results.append((name, psnr, ssim))
            total_time += elapsed
            print(f'{name:<12} {lr_size:<14} {psnr:8.2f} {ssim:8.4f} {elapsed:7.3f}s')

    avg_psnr = sum(r[1] for r in results) / len(results)
    avg_ssim = sum(r[2] for r in results) / len(results)

    print(f'{"-"*60}')
    print(f'{"Average":<12} {"":<14} {avg_psnr:8.2f} {avg_ssim:8.4f} {total_time:7.3f}s')
    print(f'{"="*60}\n')

    print(f'Set5 x{args.scale}  (Y, {args.border}px border)')
    names = '/'.join(r[0] for r in results)
    psnrs = '/'.join(f'{r[1]:.2f}' for r in results)
    ssims = '/'.join(f'{r[2]:.4f}' for r in results)
    print(f'  Per-image: {names}')
    print(f'  PSNR:      {psnrs}')
    print(f'  SSIM:      {ssims}')
    print(f'  Avg PSNR/SSIM: {avg_psnr:.2f} dB / {avg_ssim:.4f}')
    print(f'  Total time:    {total_time:.1f}s')


if __name__ == '__main__':
    main()
