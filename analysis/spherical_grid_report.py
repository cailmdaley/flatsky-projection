"""Report spherical grid tests, including finite-subpixel convergence.

The CAR prediction is the local tangent-plane parallelogram response,
averaged over orientation, multipoles and the selected latitude region.
It predicts a scalar transfer, not residual E/B mixing. The field taper is
the apodization decision both tests were run with.

Run:  bash env.sh python3 analysis/spherical_grid_report.py
          --spherical-grid-test results/fiducial/spherical_grid_test
          --spherical-grid-alias-test results/fiducial/spherical_grid_alias_test
          --apodization c2_2p5deg --out <dir>
Out:  <out>/spherical_grid_summary.json, and the figure
      <out>/spherical_grid_tests.{png,pdf}, which the paper does not include.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 8, 'text.usetex': False})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/stages'))
from spherical_grid_test import regions, hp
from _decisions import parse_apodization
import spectra as S


def load(directory):
    return [json.loads(p.read_text()) for p in sorted(directory.glob('*-statistics.json'))]


def selected(d, method, region='field1500'):
    return [x for x in d['statistics'][method] if x['region']==region]


def prediction(nside, lo, hi, region, apodization, factor=0):
    # Coarse CAR cells are pi/(3 Nside) x pi/(3 Nside).
    theta = (np.arange(3*nside)+.5)*np.pi/(3*nside)
    weight = regions(3*nside, apodization)[region].sum(axis=1)
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
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(p, 'spherical_grid_test', 'spherical_grid_alias_test')
    p.add_argument('--apodization', type=parse_apodization, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    OUT = a.out
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(a.spherical_grid_test)
    alias = load(a.spherical_grid_alias_test)
    nsides = sorted({d['settings']['nside'] for d in data})
    coarse_factor, fine_factor = sorted({d['settings']['factor'] for d in data})
    # The continuum extrapolation below assumes the fine subpixel factor doubles the coarse one.
    assert fine_factor == 2*coarse_factor
    lmax, = {d['settings']['lmax'] for d in data}
    nband = len(selected(data[0], 'car_point'))
    report = {'pixel_tests': [], 'alias_tests': [], 'convergence': []}
    for nside in nsides:
        for method in ('healpix_point', 'car_point', 'healpix_average', 'car_average',
                       'healpix_coordinate_average', 'car_coordinate_average'):
            ds = [d for d in data if d['settings']['nside']==nside and d['settings']['factor']==fine_factor]
            for band in range(nband):
                rows = [selected(d, method)[band] for d in ds]
                row = dict(nside=nside, method=method, lo=rows[0]['lo'], hi=rows[0]['hi'])
                for key in ('amplitude', 'ee', 'bb'):
                    values = np.array([r[key] for r in rows])
                    row[key] = float(values.mean())
                    row[key+'_sem'] = float(values.std(ddof=1)/np.sqrt(len(values)))
                row['bb_over_ee'] = row['bb']/row['ee']
                report['pixel_tests'].append(row)
            # Continuum estimate of the power: P(f) = P(infty)+c/f^2+O(f^-4).
            for band in range(nband):
                coarse = [selected(d, method)[band]['bb'] for d in data
                          if d['settings']['nside']==nside and d['settings']['factor']==coarse_factor]
                fine = [selected(d, method)[band]['bb'] for d in ds]
                report['convergence'].append(dict(nside=nside, method=method, band=band,
                    bb_factor4=float(np.mean(coarse)), bb_factor8=float(np.mean(fine)),
                    bb_extrapolated=float((4*np.mean(fine)-np.mean(coarse))/3)))
    for cutoff in sorted({d['settings']['input_lmax'] for d in alias}):
        for method in ('healpix_point', 'car_point', 'healpix_average', 'car_average'):
            ds = [d for d in alias if d['settings']['input_lmax']==cutoff]
            for band in range(len(selected(ds[0], method+'_alias'))):
                rows = [selected(d, method+'_alias')[band] for d in ds]
                report['alias_tests'].append(dict(input_lmax=cutoff, method=method, lo=rows[0]['lo'], hi=rows[0]['hi'],
                    ee=float(np.mean([r['ee'] for r in rows])),
                    bb=float(np.mean([r['bb'] for r in rows]))))
    apodization = a.apodization
    windows = []
    for region in ('dec0', 'dec-30', 'dec-56', 'dec-75', 'field1500'):
        ds = [d for d in data if d['settings']['nside']==nsides[0] and d['settings']['factor']==fine_factor]
        for band in range(nband):
            rows = [selected(d, 'car_average', region)[band] for d in ds]
            lo, hi = rows[0]['lo'], rows[0]['hi']
            windows.append(dict(region=region, lo=lo, hi=hi,
                amplitude=float(np.mean([r['amplitude'] for r in rows])),
                prediction=prediction(nsides[0], lo, hi, region, apodization, fine_factor),
                continuum_prediction=prediction(nsides[0], lo, hi, region, apodization)))
    report['car_windows'] = windows
    hp_windows = []
    for nside in nsides:
        wp = hp.pixwin(nside, pol=True, lmax=lmax)[1]
        wh = hp.pixwin(nside*fine_factor, pol=True, lmax=lmax)[1]
        ds = [d for d in data if d['settings']['nside']==nside and d['settings']['factor']==fine_factor]
        for region in ('fullsky', 'field1500'):
            for band in range(nband):
                rows = [selected(d, 'healpix_average', region)[band] for d in ds]
                lo, hi = rows[0]['lo'], rows[0]['hi']
                hp_windows.append(dict(nside=nside, region=region, lo=lo, hi=hi,
                    amplitude=float(np.mean([r['amplitude'] for r in rows])),
                    global_pixwin_prediction=float(np.average(wp[lo:hi]/wh[lo:hi], weights=2*np.arange(lo,hi)+1))))
    report['healpix_windows'] = hp_windows
    # These fail if point-sample leakage or the predicted CAR trend is lost.
    assert max(r['bb_over_ee'] for r in report['pixel_tests'] if r['method'].endswith('_point')) < 1e-18
    assert max(abs(r['amplitude']-r['prediction']) for r in windows) < .004
    (OUT/'spherical_grid_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {'healpix': '#6a3d9a', 'car': '#007f88'}
    for grid in ('healpix', 'car'):
        for nside, style in zip(nsides, ('-', '--')):
            rows = [r for r in report['pixel_tests'] if r['method']==grid+'_average' and r['nside']==nside]
            axes[0].plot([(r['lo']+r['hi']-1)/2 for r in rows],
                         [r['bb_over_ee'] for r in rows], style+'o', color=colors[grid],
                         label=f'{grid.upper()}, Nside={nside}' if grid=='healpix' else f'CAR, {180/(3*nside):.3f} deg')
    axes[0].set(yscale='log', xlabel=r'$\ell$', ylabel=r'$\langle W^2 B^2\rangle/\langle W^2 E^2\rangle$')
    axes[0].legend(fontsize=8, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2)
    for region, label, color in [('dec0', '0 deg', '#444444'), ('dec-30', '-30 deg', '#e69f00'),
                                 ('dec-56', '-56 deg', '#0072b2'), ('dec-75', '-75 deg', '#cc79a7')]:
        rows = [r for r in windows if r['region']==region]
        x = [(r['lo']+r['hi']-1)/2 for r in rows]
        axes[1].plot(x, [r['amplitude'] for r in rows], 'o', color=color, label=label)
        axes[1].plot(x, [r['prediction'] for r in rows], '-', color=color)
    axes[1].set(xlabel=r'$\ell$', ylabel='E cross-amplitude / input')
    axes[1].legend(fontsize=8, frameon=False, title='Declination', title_fontsize=9)
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
