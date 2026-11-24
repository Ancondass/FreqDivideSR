
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from sr_model import create_model
from test_set5 import discover_ckpt

try:
    import thop
except ImportError:
    thop = None

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None


def default_ckpt(scale):
    """ Config.CKPT_ROOT checkpoint ( None)."""
    return discover_ckpt(scale)


DEFAULT_CKPT = {s: default_ckpt(s) for s in (2, 3, 4)}

PROFILES = {
    '48':   (192, 192,  '192×192 HR (×4 LR=48×48)'),
    '720p': (720, 1280, '1280×720 HR (720p,)'),
}

BREAKDOWN_NAMES = [
    ('hpf',        'Laplacian (0, thop)'),
    ('lf_trans',   'LF Transformer (w8×5)'),
    ('hf_embed',   'HF (6→96)'),
    ('hf_norm',    'HF LayerNorm'),
    ('ode_block',  'HF NODE (RK2)'),
    ('hf_proj',    'HF (96→128)'),
    ('fusion',     ' (1×1+dw3×3+pw1×1+SE)'),
    ('recon_body', ' (4×DSResBlock)'),
    ('upsampler',  ' (PS2→refine→PS2)'),
    ('final_conv', ''),
]

SEP = '=' * 78


def lr_size(hr_h, hr_w, scale):
    """HR → LR ( 8,); (h, w,)."""
    lh, lw = hr_h // scale, hr_w // scale
    lh = max(8, lh - lh % 8)
    lw = max(8, lw - lw % 8)
    exact = (lh * scale == hr_h and lw * scale == hr_w)
    return lh, lw, exact


def load_model(scale, ckpt_path, device):
    """ test_benchmark.py / run_ablation.py."""
    Config.SCALES = [scale]
    Config.DEVICE = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ablation = ckpt.get('ablation') if isinstance(ckpt, dict) else None
    model = create_model(scales=[scale], ablation=ablation)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    if any(k.startswith('yolo11_backbone.') for k in state):
        state = {k.replace('yolo11_backbone.', 'global_branch.'): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f' [Warn] {len(missing)}: {missing[:3]}...')
    if unexpected:
        print(f' [Warn] {len(unexpected)}: {unexpected[:3]}...')
    model.eval()
    return model, ablation


def count_flops(module, x):
    """ {'macs','thop_flops','counter'} (GFLOPs); None."""
    res = {'macs': None, 'thop_flops': None, 'counter': None}

    if thop is not None:
        try:
            with torch.no_grad():
                macs, _ = thop.profile(module, inputs=(x,), verbose=False)
            res['macs'] = macs / 1e9
            res['thop_flops'] = macs * 2 / 1e9          # 1 MAC = 2 FLOPs
        except Exception as e:                          # noqa: BLE001
            print(f' [Warn] thop ({type(e).__name__}): {e}')

    if FlopCounterMode is not None:
        try:
            with torch.no_grad():
                with FlopCounterMode(display=False) as fc:
                    module(x)
            res['counter'] = fc.get_total_flops() / 1e9
        except Exception as e:                          # noqa: BLE001
            print(f' [Warn] FlopCounterMode ({type(e).__name__}): {e}')

    return res


def pick_flops(res):
    if not res:
        return None, 'n/a'
    if res.get('macs') is not None:
        return res['macs'], 'thop MACs'
    if res.get('counter') is not None:
        return res['counter'] / 2.0, 'counter/2 ≈ MACs'
    return None, 'n/a'


def capture_inputs(model, x, names):
    """, ()."""
    buf = {}
    handles = []

    def make_hook(key):
        def hook(module, inp, out):
            if key not in buf:
                buf[key] = inp[0].detach()
        return hook

    for key, _ in names:
        mod = model.get_submodule(key)
        handles.append(mod.register_forward_hook(make_hook(key)))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return buf


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def measure_latency(model, x, device, warmup, iters, self_ensemble, scale):
    if self_ensemble:
        from test_set5 import self_ensemble_forward

        def run():
            self_ensemble_forward(model, x, scale=scale)
    else:
        def run():
            model(x)

    with torch.inference_mode():
        for _ in range(warmup):
            run()
        sync(device)
        ts = []
        for _ in range(iters):
            sync(device)
            t0 = time.perf_counter()
            run()
            sync(device)
            ts.append((time.perf_counter() - t0) * 1000.0)
    t = torch.tensor(ts)
    return t.mean().item(), t.std().item(), t.median().item()


def measure_peak_memory(model, x, device):
    if device.type != 'cuda':
        return None
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(x)
    sync(device)
    return torch.cuda.max_memory_allocated(device) / 1e9


def main():
    ap = argparse.ArgumentParser(description=' 4 / ( §4.6)')
    ap.add_argument('--scale', type=int, default=4, choices=[2, 3, 4])
    ap.add_argument('--ckpt', default=None, help=' ()')
    ap.add_argument('--device', default=None, help='cuda / cuda:0 / cpu ()')
    ap.add_argument('--profiles', nargs='+', default=['48', '720p'], choices=list(PROFILES),
                    help=' ( 48 720p)')
    ap.add_argument('--breakdown-profile', default='48', choices=list(PROFILES),
                    help=' ( 48)')
    ap.add_argument('--no-breakdown', action='store_true', help='')
    ap.add_argument('--warmup', type=int, default=20,
                    help=' ( 20, measure_step_sweep.py)')
    ap.add_argument('--iters', type=int, default=100,
                    help=' ( 100, measure_step_sweep.py)')
    ap.add_argument('--self-ensemble', action='store_true',
                    help=' 8 ( 8)')
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else Config.DEVICE
    if device.type == 'cpu':
        args.iters = min(args.iters, 20)
        print(f'[Warn] CPU, ; iters {args.iters}')

    ckpt_path = args.ckpt or default_ckpt(args.scale)
    print(SEP)
    print(f': {device}')
    print(f': x{args.scale}')
    print(f': {ckpt_path}')
    print(f'thop: {"" if thop is not None else " ( FlopCounterMode)"}')
    print(f': batch=1, FP32, inference_mode, {args.warmup} / {args.iters} ')
    print(SEP)

    if not os.path.exists(ckpt_path):
        print(f'[Error]: {ckpt_path}')
        return 1

    model, ablation = load_model(args.scale, ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f'    Total: {n_params:,}  ({n_params / 1e6:.2f} M)')
    print(f' ablation: {sorted(ablation) if ablation else " ()"}')
    top = []
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n:
            top.append((name, n))
    top.sort(key=lambda kv: -kv[1])
    for name, n in top[:10]:
        print(f'      {name:<16} {n:>10,}  ({n / n_params * 100:5.1f}%)')

    # ---------------- [2] FLOPs ----------------
    print(f' {"":<7}{"(LR)":<12}{"(HR)":<12}{"MACs(thop)":>11}'
          f'{"MACs(counter/2)":>16}{"FLOPs(counter)":>15}')
    print('    ' + '-' * 70)
    flops_table, size_table = {}, {}
    for key in args.profiles:
        hr_h, hr_w, desc = PROFILES[key]
        lh, lw, exact = lr_size(hr_h, hr_w, args.scale)
        size_table[key] = (lh, lw)
        if not exact:
            print(f' [Note] {key}: x{args.scale}, LR {lh}×{lw} '
                  f'(HR {lh * args.scale}×{lw * args.scale}, {hr_h}×{hr_w})')
        x = torch.randn(1, 3, lh, lw, device=device)
        res = count_flops(model, x)
        flops_table[key] = res

        def g(v):
            return f'{v:.1f} G' if v is not None else ''

        in_s = f'{lh}×{lw}'
        out_s = f'{lh * args.scale}×{lw * args.scale}'
        macs_c = res['counter'] / 2.0 if res.get('counter') is not None else None
        print(f'    {key:<7}{in_s:<12}{out_s:<12}{g(res["macs"]):>11}'
              f'{g(macs_c):>16}{g(res["counter"]):>15}')
        print(f'           {desc}')

    if '48' in flops_table:
        f48, src48 = pick_flops(flops_table['48'])
        if f48 is not None:
            print(f' -> 48 {f48:.1f} G MACs ({src48})'
                  ' 4 46.7 G (), ')

    if not args.no_breakdown:
        key = args.breakdown_profile
        lh, lw = size_table.get(key) or lr_size(*PROFILES[key][:2], args.scale)[:2]
        x = torch.randn(1, 3, lh, lw, device=device)
        print(f'\n[3] FLOPs ( {key}: {lh}×{lw} → '
              f'{lh * args.scale}×{lw * args.scale})')
        buf = capture_inputs(model, x, BREAKDOWN_NAMES)
        total, _ = pick_flops(flops_table.get(key))
        parts = []
        for name, label in BREAKDOWN_NAMES:
            if name not in buf:
                continue
            r = count_flops(model.get_submodule(name), buf[name])
            f, _src = pick_flops(r)
            parts.append((name, label, f))
        print(f' {"":<14}{"FLOPs":>10}{"":>8} ')
        print('    ' + '-' * 78)
        for name, label, f in sorted(parts, key=lambda t: -(t[2] or 0)):
            fs = f'{f:.1f} G' if f is not None else ''
            share = f'{f / total * 100:6.1f}%' if (f is not None and total) else ' '
            print(f'    {name:<14}{fs:>10}{share:>8}  {label}')
        print('    ' + '-' * 78)
        s = sum(f for _, _, f in parts if f is not None)
        line = f' {"":<14}{s:>10.1f} G'
        if total:
            line += (f' ( {total:.1f} G, {abs(s - total) / total * 100:.1f}%'
                     ' bicubic)')
        print(line)

    lat = {}
    for key in args.profiles:
        lh, lw = size_table[key]
        x = torch.randn(1, 3, lh, lw, device=device)
        mean, std, med = measure_latency(model, x, device, args.warmup, args.iters,
                                         False, args.scale)
        peak = measure_peak_memory(model, x, device)
        lat[key] = (mean, peak)
        mem = f' {peak:.2f} GB' if peak is not None else ' (CPU)'
        print(f'    {key:<6}{lh}×{lw} → {lh * args.scale}×{lw * args.scale}: '
              f'{mean:8.2f} ± {std:5.2f} ms ( {med:8.2f}) {mem}')
        if args.self_ensemble:
            m8, s8, md8 = measure_latency(model, x, device, max(1, args.warmup // 5),
                                          max(1, args.iters // 5), True, args.scale)
            print(f' 8: {m8:8.2f} ± {s8:5.2f} ms ( {md8:8.2f})')

    print(f'\n[5] 4 ( {n_params / 1e6:.2f} M)')
    for key in args.profiles:
        f, src = pick_flops(flops_table.get(key))
        if f is None:
            continue
        mean = lat[key][0]
        lh, lw = size_table[key]
        px = (lh * args.scale) * (lw * args.scale)
        print(f'    | **Ours** | **{n_params / 1e6:.2f}** | **{f:.1f}** ({key}, MACs) '
              f'| **{mean:.1f}** |')
        print(f' {key}: {lh * args.scale}×{lw * args.scale}, '
              f'{f * 1e3 / px:.3f} MMACs/, {src}')

    REF_720P = [('EDSR', 2890.0), ('RCAN', 5200.0), ('CARN', 90.9), ('LAPAR-A', 94.0),
                ('SwinIR-light', 49.6), ('IMDN', 40.9), ('RFDN', 30.9), ('PAN', 27.0),
                ('SMSR', 23.4)]
    print('      EDSR 2890 G | RCAN 5200 G | CARN 90.9 G | IMDN 40.9 G | RFDN 30.9 G')
    print('      PAN 27.0 G  | SMSR 23.4 G | LAPAR-A 94.0 G | SwinIR-light 49.6 G')

    ours_pp = None
    for key in ('720p', '48'):
        if key in flops_table:
            f, _ = pick_flops(flops_table[key])
            if f is not None:
                lh, lw = size_table[key]
                ours_pp = f * 1e3 / (lh * lw)
                src_key = key
                break
    if ours_pp:
        print(f'\n ( LR MACs; 720p LR=320×180):')
        print(f' {"Ours":<14}{ours_pp:9.3f} MMACs/LR-px ( {src_key})')
        for nm, g_ref in REF_720P:
            pp = g_ref * 1e3 / 57600
            print(f'      {nm:<14}{pp:9.3f} MMACs/LR-px  '
                  f'(Ours {ours_pp / pp:5.1f})')
    print(SEP)
    return 0


if __name__ == '__main__':
    sys.exit(main())
