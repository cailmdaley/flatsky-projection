"""Report spherical grid tests, including finite-subpixel convergence.

Reads lc's gathered products only. Run under bash env.sh.
The CAR prediction is the local tangent-plane parallelogram response,
averaged over orientation, multipoles and the selected latitude region.
It predicts a scalar transfer, not residual E/B mixing.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/stages'))
from spherical_grid_test import regions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/figures'
OUT.mkdir(exist_ok=True)


def load(name, count):
    paths = sorted((ROOT/'results/highell'/name).glob('*-statistics.json'))
    assert len(paths) == count, (name, len(paths), count)
    return [json.loads(p.read_text()) for p in paths]


def selected(d, method, region='field1500'):
    return [x for x in d['statistics'][method] if x['region']==region]


def prediction(nside, lo, hi, region, factor=0):
    # Coarse CAR cells are pi/(3 Nside) x pi/(3 Nside).
    theta = (np.arange(384)+.5)*np.pi/384
    weight = regions(384)[region].sum(axis=1)
    keep = weight>0
    theta, weight = theta[keep], weight[keep]
    chi = (np.arange(180)+.5)*np.pi/180
    ell = np.arange(lo, hi)
    k = np.sqrt(ell*(ell+1))[:, None, None]*np.pi/(3*nside)/2
    x = k*np.cos(chi)[None, None, :]
    y = k*np.sin(theta)[None, :, None]*np.sin(chi)[None, None, :]
    w = np.sinc(x/np.pi)*np.sinc(y/np.pi)
    if factor:
        w /= np.sinc(x/(np.pi*factor))*np.sinc(y/(np.pi*factor))
    byell = np.average(w.mean(axis=-1), axis=1, weights=weight)
    return float(np.average(byell, weights=2*ell+1))


def main():
    data = load('spherical_grid_test', 12)
    alias = load('spherical_grid_alias_test', 3)
    report = {'pixel_tests': [], 'alias_tests': [], 'convergence': []}
    for nside in (128, 256):
        for method in ('healpix_point', 'car_point', 'healpix_average', 'car_average',
                       'healpix_coordinate_average', 'car_coordinate_average'):
            ds = [d for d in data if d['settings']['nside']==nside and d['settings']['factor']==8]
            for band in range(4):
                rows = [selected(d, method)[band] for d in ds]
                row = dict(nside=nside, method=method, lo=rows[0]['lo'], hi=rows[0]['hi'])
                for key in ('amplitude', 'ee', 'bb'):
                    values = np.array([r[key] for r in rows])
                    row[key] = float(values.mean())
                    row[key+'_sem'] = float(values.std(ddof=1)/np.sqrt(len(values)))
                row['bb_over_ee'] = row['bb']/row['ee']
                report['pixel_tests'].append(row)
            # Continuum estimate of the power: P(f) = P(infty)+c/f^2+O(f^-4).
            for band in range(4):
                coarse = [selected(d, method)[band]['bb'] for d in data
                          if d['settings']['nside']==nside and d['settings']['factor']==4]
                fine = [selected(d, method)[band]['bb'] for d in ds]
                report['convergence'].append(dict(nside=nside, method=method, band=band,
                    bb_factor4=float(np.mean(coarse)), bb_factor8=float(np.mean(fine)),
                    bb_extrapolated=float((4*np.mean(fine)-np.mean(coarse))/3)))
    for method in ('healpix_point', 'car_point', 'healpix_average', 'car_average'):
        for band in range(4):
            rows = [selected(d, method+'_alias')[band] for d in alias]
            report['alias_tests'].append(dict(method=method, lo=rows[0]['lo'], hi=rows[0]['hi'],
                ee=float(np.mean([r['ee'] for r in rows])),
                bb=float(np.mean([r['bb'] for r in rows]))))
    windows = []
    for region in ('dec0', 'dec-30', 'dec-56', 'dec-75', 'field1500'):
        ds = [d for d in data if d['settings']['nside']==128 and d['settings']['factor']==8]
        for band in range(4):
            rows = [selected(d, 'car_average', region)[band] for d in ds]
            lo, hi = rows[0]['lo'], rows[0]['hi']
            windows.append(dict(region=region, lo=lo, hi=hi,
                amplitude=float(np.mean([r['amplitude'] for r in rows])),
                prediction=prediction(128, lo, hi, region, 8),
                continuum_prediction=prediction(128, lo, hi, region)))
    report['car_windows'] = windows
    (OUT/'spherical_grid_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {'healpix': '#6a3d9a', 'car': '#007f88'}
    for grid in ('healpix', 'car'):
        for nside, style in ((128, '-'), (256, '--')):
            rows = [r for r in report['pixel_tests'] if r['method']==grid+'_average' and r['nside']==nside]
            axes[0].plot([(r['lo']+r['hi']-1)/2 for r in rows],
                         [r['bb_over_ee'] for r in rows], style+'o', color=colors[grid],
                         label=f'{grid.upper()}, Nside={nside}' if grid=='healpix' else f'CAR, {180/(3*nside):.3f} deg')
    axes[0].set(yscale='log', xlabel=r'$\ell$', ylabel=r'$\langle W^2 B^2\rangle/\langle W^2 E^2\rangle$')
    axes[0].legend(fontsize=8, frameon=False)
    for region, label, color in [('dec0', '0 deg', '#444444'), ('dec-30', '-30 deg', '#e69f00'),
                                 ('dec-56', '-56 deg', '#0072b2'), ('dec-75', '-75 deg', '#cc79a7')]:
        rows = [r for r in windows if r['region']==region]
        x = [(r['lo']+r['hi']-1)/2 for r in rows]
        axes[1].plot(x, [r['amplitude'] for r in rows], 'o', color=color, label=label)
        axes[1].plot(x, [r['prediction'] for r in rows], '-', color=color)
    axes[1].set(xlabel=r'$\ell$', ylabel='E cross-amplitude / input')
    axes[1].legend(fontsize=8, frameon=False, title='Declination')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/'spherical_grid_tests.png', dpi=170)
    fig.savefig(OUT/'spherical_grid_tests.pdf')
    for r in report['pixel_tests']:
        if r['lo']==192:
            print(r)
    print('CAR window largest absolute amplitude error:', max(abs(r['amplitude']-r['prediction']) for r in windows))
    print('Alias:', [r for r in report['alias_tests'] if r['lo']==192])


if __name__=='__main__':
    main()
