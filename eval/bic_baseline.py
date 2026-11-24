# -*- coding: utf-8 -*-
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from test_set5 import calc_psnr_y, calc_ssim_y


def to_tensor(im):
    a = np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def find_lr(lr_dir, name, scale):
    for pat in ('%s.png', '%sx%d.png', '%s_x%d.png', '%sX%d.png',
                '%s.%d.png', '%s*.png'):
        hit = sorted(glob.glob(os.path.join(lr_dir, pat % (name, scale) if pat.count('%') == 2
                                            else pat % name)))
        if hit:
            return hit[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help=' HR LR_bicubic')
    ap.add_argument('--scale', '-s', type=int, default=4)
    ap.add_argument('--border', type=int, default=4)
    ap.add_argument('--mod-crop', action='store_true', help='HR scale EDSR ')
    ap.add_argument('--images', default='', help=' img_018,img_025')
    args = ap.parse_args()

    hr_dir = os.path.join(args.data, 'HR')
    lr_dir = os.path.join(args.data, 'LR_bicubic', 'X%d' % args.scale)
    want = [x for x in args.images.split(',') if x]

    rows = []
    for hr_p in sorted(glob.glob(os.path.join(hr_dir, '*.png'))):
        name = os.path.splitext(os.path.basename(hr_p))[0]
        if want and name not in want:
            continue
        lr_p = find_lr(lr_dir, name, args.scale)
        if lr_p is None:
            print('!! LR:', name)
            continue
        hr = Image.open(hr_p).convert('RGB')
        lr = Image.open(lr_p).convert('RGB')
        if args.mod_crop:
            w, h = hr.size
            hr = hr.crop((0, 0, w - w % args.scale, h - h % args.scale))
        bic = lr.resize(hr.size, Image.BICUBIC)
        hr_t, bic_t = to_tensor(hr), to_tensor(bic)
        psnr = float(calc_psnr_y(bic_t, hr_t, border=args.border))
        ssim = float(calc_ssim_y(bic_t, hr_t, border=args.border))
        rows.append((name, psnr, ssim))
        print('%-16s %8.2f dB  %.4f' % (name, psnr, ssim))

    if rows:
        print('=' * 44)
        print('%-16s %8.2f dB  %.4f  (n=%d, border=%d%s)'
              % ('Average', sum(r[1] for r in rows) / len(rows),
                 sum(r[2] for r in rows) / len(rows), len(rows), args.border,
                 ', mod-crop' if args.mod_crop else ''))
        print(' Urban100 x%d: 23.14 dB / 0.6577' % args.scale)


if __name__ == '__main__':
    main()
