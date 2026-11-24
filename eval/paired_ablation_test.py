import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ablation import LOGS_ABL, VARIANTS

TEST_ORDER = ['only_lf', 'only_hf', 'node_to_cnn', 'trans_to_cnn', 'both_to_cnn',
              'no_highway', 'simple_fusion', 'no_se', 'no_lap']
N_BOOT = 20000
SEED = 20260914


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def _betacf(a, b, x, itmax=300, eps=3e-14):
    """, Beta (Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """ Beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """ P(T > t), T ~ t(df)."""
    if df <= 0 or not np.isfinite(t):
        return float('nan')
    x = df / (df + t * t)
    two_sided = _betai(0.5 * df, 0.5, x)
    return two_sided / 2.0 if t >= 0 else 1.0 - two_sided / 2.0


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _rankdata(a):
    """ (), scipy.stats.rankdata."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind='mergesort')
    sa = a[order]
    ranks = np.empty(a.size, dtype=float)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def wilcoxon_greater(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = d.size
    if n < 6:
        return float('nan'), float('nan'), n
    r = _rankdata(np.abs(d))
    w_pos = float(r[d > 0].sum())
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = float((counts ** 3 - counts).sum()) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie
    if var <= 0:
        return float('nan'), w_pos, n
    z = (w_pos - mean - 0.5) / math.sqrt(var)
    return 1.0 - _phi(z), w_pos, n


def paired_stats(full_x, var_x):
    """ full vs variant. d = full − variant, H1: mean(d) > 0."""
    d = np.asarray(full_x, float) - np.asarray(var_x, float)
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float('nan')
    se = sd / math.sqrt(n) if n > 1 else float('nan')
    t = mean / se if se and np.isfinite(se) and se > 0 else float('nan')
    p_t = t_sf(t, n - 1)
    p_w, w_pos, n_eff = wilcoxon_greater(d)
    return dict(n=n, mean=mean, sd=sd, se=se, t=t, p_t=p_t, p_w=p_w, w_pos=w_pos,
                n_eff=n_eff, cohen=mean / sd if sd and sd > 0 else float('nan'),
                worse=int((d > 0).sum()), equal=int((d == 0).sum()),
                better=int((d < 0).sum()), d=d)


def one_sample_greater(x):
    """ H1: mean(x) > 0."""
    x = np.asarray(x, float)
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float('nan')
    se = sd / math.sqrt(n) if n > 1 else float('nan')
    t = mean / se if se and np.isfinite(se) and se > 0 else float('nan')
    p_t = t_sf(t, n - 1)
    p_w, w_pos, n_eff = wilcoxon_greater(x)
    return dict(n=n, mean=mean, sd=sd, se=se, t=t, p_t=p_t, p_w=p_w, w_pos=w_pos,
                n_eff=n_eff, cohen=mean / sd if sd and sd > 0 else float('nan'),
                pos=int((x > 0).sum()), zero=int((x == 0).sum()),
                neg=int((x < 0).sum()), x=x)


def bootstrap_ci(x, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2, 100.0 * (1.0 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvals):
    """HolmBonferroni, p."""
    arr = np.asarray(pvals, float)
    order = np.argsort(arr)
    m = arr.size
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, arr[idx] * (m - rank))
        running = max(running, val)
        adj[idx] = running
    return adj


def _fmt_p(p):
    if not np.isfinite(p):
        return '   n/a  '
    if p < 1e-4:
        return '<1e-4  '
    return f'{p:7.4f}'


def _stars(p):
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def load_per_image(base_dir, dataset, scale, self_ensemble, variants):
    """ CSV: <base>/<variant>/**/per_image_{dataset}_x{scale}[_se].csv"""
    tag = '_se' if self_ensemble else ''
    data, paths = {}, {}
    for v in variants:
        pat = os.path.join(base_dir, v, '**', f'per_image_{dataset}_x{scale}{tag}.csv')
        hits = sorted(glob.glob(pat, recursive=True))
        if not hits:
            print(f'[SKIP] {v:16s} CSV ( run_ablation.py... --dump-csv)')
            continue
        rows = {}
        with open(hits[-1], newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                rows[row['image']] = (float(row['psnr']), float(row['ssim']))
        if not rows:
            print(f'[SKIP] {v:16s} CSV: {hits[-1]}')
            continue
        data[v], paths[v] = rows, hits[-1]
        print(f'[CSV ] {v:16s} {os.path.relpath(hits[-1], base_dir)}  ({len(rows)} images)')
    return data, paths


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir', default=LOGS_ABL, help=f'logs_abl ( {LOGS_ABL})')
    parser.add_argument('--scale', '-s', type=int, default=4)
    parser.add_argument('--dataset', '-d', default='Urban100')
    parser.add_argument('--self-ensemble', action='store_true',
                        help=' CSV ( _se, 2)')
    parser.add_argument('--variants', nargs='+', default=None, help='')
    parser.add_argument('--alpha', type=float, default=0.05, help=' ( 0.05)')
    parser.add_argument('--out', default=None, help=' markdown ')
    args = parser.parse_args()

    variants = args.variants or list(VARIANTS)
    if 'full' not in variants:
        variants = ['full'] + [v for v in variants if v != 'full']

    print('=' * 78)
    print(f' {args.dataset} x{args.scale}, '
          f'{" ON" if args.self_ensemble else " OFF"} '
          f'(Y, 4px)')
    print(f'CSV: {args.dir}')
    print('=' * 78)

    data, paths = load_per_image(args.dir, args.dataset, args.scale,
                                 args.self_ensemble, variants)
    if 'full' not in data:
        print(': python run_ablation.py --summary --scale %d --dataset %s%s --dump-csv'
              % (args.scale, args.dataset, ' --self-ensemble' if args.self_ensemble else ''))
        return

    common = sorted(set.intersection(*[set(d) for d in data.values()]))
    if len(common) < 5:
        print(f'[Error] ({len(common)}),.')
        return
    print(f'\n: {len(common)} ()\n')

    full_p = np.array([data['full'][k][0] for k in common])
    full_s = np.array([data['full'][k][1] for k in common])
    psnr = {v: np.array([data[v][k][0] for k in common]) for v in data}
    ssim = {v: np.array([data[v][k][1] for k in common]) for v in data}

    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(s)

    todo = [v for v in TEST_ORDER if v in data and v != 'full']
    pvals = []
    stats = {}
    for v in todo:
        stats[v] = paired_stats(full_p, psnr[v])
        pvals.append(stats[v]['p_t'])
    p_holm = holm(pvals) if pvals else np.array([])

    emit('─' * 78)
    emit('[A] vs full PSNR (d_i = PSNR_full,i − PSNR_variant,i, H1: mean d > 0)')
    emit('─' * 78)
    emit(f'{"variant":<16}{"mean ΔPSNR":>11}{"Δstd":>8}{"t":>8}{"p_t":>8}{"p_Wilcox":>10}'
         f'{"p_Holm":>9}{"worse/tie/better":>19}')
    emit(f'{"":16}{"(dB)":>11}{"":8}{"":8}{"":8}{"":10}{"":9}{"":19}')
    for i, v in enumerate(todo):
        st = stats[v]
        emit(f'{v:<16}{-st["mean"]:>11.3f}{st["sd"]:>8.3f}{st["t"]:>8.2f}'
             f'{_fmt_p(st["p_t"]):>8}{_fmt_p(st["p_w"]):>10}{_fmt_p(p_holm[i]):>9}'
             f'{st["worse"]:>8}/{st["equal"]}/{st["better"]:<8}{_stars(p_holm[i])}')
    emit('')
    emit(': mean ΔPSNR = full ; "worse/tie/better" ( full);')
    emit(' p_t t, p_Wilcox Wilcoxon ;')
    emit(' p_Holm 9 Holm ().')

    need = ['trans_to_cnn', 'node_to_cnn', 'both_to_cnn']
    if all(v in data for v in need):
        sigma = (psnr['trans_to_cnn'] + psnr['node_to_cnn']
                 - psnr['both_to_cnn'] - psnr['full'])
        st_s = one_sample_greater(sigma)
        lo, hi = bootstrap_ci(sigma)
        d_b = paired_stats(full_p, psnr['both_to_cnn'])
        d_l = paired_stats(full_p, psnr['trans_to_cnn'])
        d_h = paired_stats(full_p, psnr['node_to_cnn'])
        emit('')
        emit('─' * 78)
        emit('[B] σ = ΔB − ΔL − ΔH (H1: mean σ > 0,)')
        emit('─' * 78)
        emit(f' ΔB (both_to_cnn) = {d_b["mean"]:.4f} dB ( t = {d_b["t"]:.2f}, '
             f'p = {_fmt_p(d_b["p_t"]).strip()})')
        emit(f' ΔL (trans_to_cnn)= {d_l["mean"]:.4f} dB ( t = {d_l["t"]:.2f}, '
             f'p = {_fmt_p(d_l["p_t"]).strip()})')
        emit(f' ΔH (node_to_cnn) = {d_h["mean"]:.4f} dB ( t = {d_h["t"]:.2f}, '
             f'p = {_fmt_p(d_h["p_t"]).strip()})')
        emit(f' σ () = {st_s["mean"]:.4f} dB ± {st_s["sd"]:.4f} (std)')
        emit(f' 95% CI = [{lo:.4f}, {hi:.4f}] dB ({N_BOOT})')
        emit(f' t = t({st_s["n"] - 1}) = {st_s["t"]:.3f}, '
             f'p = {_fmt_p(st_s["p_t"]).strip()} {_stars(st_s["p_t"])}')
        emit(f' Wilcoxon = W+ = {st_s["w_pos"]:.1f}, '
             f'p = {_fmt_p(st_s["p_w"]).strip()} {_stars(st_s["p_w"])}')
        emit(f'    Cohen\'s d       = {st_s["cohen"]:.3f}')
        emit(f' σ_i > 0 = {st_s["pos"]}/{st_s["n"]} '
             f'(=0: {st_s["zero"]}, <0: {st_s["neg"]})')
        emit(f' σ = 5% {np.percentile(sigma, 5):.3f} | 50% '
             f'{np.percentile(sigma, 50):.3f} | 95% {np.percentile(sigma, 95):.3f}')
        emit(f': σ = {psnr["trans_to_cnn"].mean():.4f} + '
             f'{psnr["node_to_cnn"].mean():.4f} − {psnr["both_to_cnn"].mean():.4f} − '
             f'{full_p.mean():.4f} = {st_s["mean"]:.4f} dB ()')
        verdict = ('σ 0 →, '
                   if np.isfinite(st_s['p_t']) and st_s['p_t'] < args.alpha
                   else 'σ → "()", ')
        emit(f': {verdict}')
    else:
        missing = [v for v in need if v not in data]
        emit('')
        emit(f'[B] σ: {missing} CSV')

    s_pvals, s_stats = [], {}
    for v in todo:
        s_stats[v] = paired_stats(full_s, ssim[v])
        s_pvals.append(s_stats[v]['p_t'])
    s_holm = holm(s_pvals) if s_pvals else np.array([])
    emit('')
    emit('─' * 78)
    emit('[C] vs full SSIM ()')
    emit('─' * 78)
    emit(f'{"variant":<16}{"mean ΔSSIM":>12}{"Δstd":>9}{"t":>8}{"p_t":>8}{"p_Holm":>9}'
         f'{"worse/tie/better":>19}')
    for i, v in enumerate(todo):
        st = s_stats[v]
        emit(f'{v:<16}{-st["mean"]:>12.5f}{st["sd"]:>9.5f}{st["t"]:>8.2f}'
             f'{_fmt_p(st["p_t"]):>8}{_fmt_p(s_holm[i]):>9}'
             f'{st["worse"]:>8}/{st["equal"]}/{st["better"]:<8}{_stars(s_holm[i])}')
    if all(v in data for v in need):
        sig_s = (ssim['trans_to_cnn'] + ssim['node_to_cnn']
                 - ssim['both_to_cnn'] - ssim['full'])
        st_ss = one_sample_greater(sig_s)
        emit('')
        emit(f' SSIM σ_SSIM = {st_ss["mean"]:.6f} ± {st_ss["sd"]:.6f}, '
             f't = {st_ss["t"]:.3f}, p = {_fmt_p(st_ss["p_t"]).strip()} '
             f'{_stars(st_ss["p_t"])}, σ>0 {st_ss["pos"]}/{st_ss["n"]}')

    try:
        from scipy import stats as _sp
    except Exception:
        _sp = None
    if _sp is not None and todo:
        v0 = todo[0]
        d0 = stats[v0]['d']
        t0, p0 = _sp.ttest_rel(full_p, psnr[v0], alternative='greater')
        emit('')
        emit(f' [scipy] {v0}: t = {t0:.3f}, p = {p0:.6g} '
             f'(: t = {stats[v0]["t"]:.3f}, p = {stats[v0]["p_t"]:.6g})')
    else:
        emit('')
        emit(' [] scipy, (t Beta,')
        emit(' Wilcoxon).')

    emit('')
    emit(': p "", /;')
    emit(' ( no_se/no_lap 0.030.04 dB), "')
    emit(' ", "".')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(f'# {args.dataset} x{args.scale}'
                     f'{" ( ON)" if args.self_ensemble else ""}\n\n')
            fh.write('```\n' + '\n'.join(lines) + '\n```\n')
        print(f'\n: {args.out}')


if __name__ == '__main__':
    main()
