"""Diagnostic figures for estimator response, independent means and B floor."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spectra as S
import paper_style as PS

ROOT = Path('/sptlocal/user/cailmdaley/pol_project')
OUT = S.ROOT/'analysis/figures/residual_anomalies'


def plot_ring_controls():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    for i, proj in enumerate(('ProjSTG', 'ProjZEA')):
        data = [np.load(S.ROOT/'results/highell/residual_ring_test'/f'rings_{proj}_seed{s:04d}.npz') for s in range(2)]
        for ax, (lo, hi) in zip(axes[i], ((100, 1000), (1000, 4800))):
            sel = (S.ELL >= lo) & (S.ELL < hi)
            for key, style, label in [('n8192_r8', '-', r'$N_{side}=8192$, $8$ samples/axis'),
                                      ('n4096_r8', '--', r'$N_{side}=4096$, $8$ samples/axis'),
                                      ('n4096_r16', ':', r'$N_{side}=4096$, $16$ samples/axis')]:
                values = np.array([d[key][3, sel]/d[key][0, sel] for d in data])
                scale = 1e6 if proj == 'ProjSTG' else 1e4
                mean = scale*values.mean(0)
                error = scale*values.std(0, ddof=1)/np.sqrt(len(values))
                ax.plot(S.ELL[sel], mean, style, color=PS.PROJ_COLORS[proj], label=label)
                ax.fill_between(S.ELL[sel], mean-error, mean+error, color=PS.PROJ_COLORS[proj], alpha=.08)
            ax.set_ylabel(r'$10^6 BB/EE$' if proj == 'ProjSTG' else r'$10^4 BB/EE$')
            ax.text(.04, .94, PS.PROJ_LABEL[proj], transform=ax.transAxes, va='top')
            ax.set_xlabel(r'$\ell$')
            ax.grid(alpha=.15)
        for d in data:
            d.close()
    axes[0, 1].legend(fontsize=8, loc='center left')
    fig.savefig(OUT/'ring_controls.png', dpi=160)
    fig.savefig(OUT/'ring_controls.pdf')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = np.load(ROOT/'pipeline_windows_production.npz')
    response = np.load(ROOT/'residual_estimator_response.npz')
    data = np.load(ROOT/'residual_independent_means.npz')
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 9, 'text.usetex': False,
                         'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False})
    for limits, name in [((100, 1000), 'low_ell'), ((1000, 4800), 'high_ell')]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True,
                                 layout='constrained')
        sel = (S.ELL >= limits[0]) & (S.ELL < limits[1])
        ell = S.ELL[sel]
        for j, proj in enumerate(('ProjSFL', 'ProjSTG', 'ProjZEA')):
            for i, spec in enumerate(('TT', 'EE')):
                ax = axes[i, j]
                model = report[f'field_{spec}_{proj}_windowed'][sel]
                old = data[f'{proj}_{spec}_production'][:, sel]/model-1
                new = data[f'{proj}_{spec}_independent'][:, sel]/model-1
                ax.plot(ell, 100*old.mean(0), '.', color='.72', ms=3, label='production 20')
                ax.errorbar(ell, 100*new.mean(0), yerr=100*new.std(0, ddof=1)/np.sqrt(len(new)),
                            fmt=PS.PROJ_MARKERS[proj], ms=3, elinewidth=.6,
                            color=PS.PROJ_COLORS[proj], label='independent 260')
                ax.plot(ell, 100*(response[f'{proj}_{spec}_estimator'][sel]/model-1),
                        '--', color='.45', lw=1.2, label='radial estimator response')
                ax.plot(ell, 100*(response[f'{proj}_{spec}_spherical'][sel]/model-1),
                        color='.1', lw=1.2, label=r'plus $k=\ell+1/2$')
                ax.axhline(0, lw=.5, color='.7')
                ax.text(.97, .94, f'{PS.PROJ_LABEL[proj]} {spec}', ha='right', va='top', transform=ax.transAxes)
                ax.grid(alpha=.15)
                if j == 0:
                    ax.set_ylabel('Residual [%]')
                if i == 1:
                    ax.set_xlabel(r'$\ell$')
        axes[0, 0].legend(fontsize=8, loc='lower left')
        axes[0, 0].set_ylim((-6, 4) if name == 'low_ell' else (-3, 2))
        fig.savefig(OUT/f'{name}.png', dpi=160)
        fig.savefig(OUT/f'{name}.pdf')
        plt.close(fig)

    # Pure-E controls: BB normalized to periodic E at each bin; signed values
    # remain visible rather than imposing a positivity cut on tiny floors.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for proj in S.PROJS:
        paths = sorted((S.ROOT/'results/highell/residual_pure_e_test').glob(f'pure_e_{proj}_seed*.npz'))
        if len(paths) != 4:
            raise ValueError('Pure-E control ensemble is incomplete')
        d = [np.load(path) for path in paths]
        ee = np.array([z['cut6000_x1_periodic'][0] for z in d])
        for ax, factor in zip(axes, (1, 2)):
            bb = np.array([z[f'cut6000_x{factor}_decoupled'][3] for z in d])
            y = (bb/ee).mean(0)*(1e6 if factor == 1 else 1e12)
            ax.plot(S.ELL, y, color=PS.PROJ_COLORS[proj], marker=PS.PROJ_MARKERS[proj],
                    ms=2, lw=1, label=PS.PROJ_LABEL[proj])
            ax.set_xlim(1000, 4800)
            ax.set_ylim(-1, 20)
            ax.set_xlabel(r'$\ell$')
            ax.grid(alpha=.15)
        for z in d:
            z.close()
    axes[0].set_ylabel(r'$10^6 BB/EE_{\rm periodic}$')
    axes[1].set_ylabel(r'$10^{12} BB/EE_{\rm periodic}$')
    axes[0].text(.03, .94, 'Native mask products', transform=axes[0].transAxes, va='top')
    axes[1].text(.03, .94, 'Twice-finer mask products', transform=axes[1].transAxes, va='top')
    axes[0].legend(loc='lower left', fontsize=9)
    fig.savefig(OUT/'pure_e.png', dpi=160)
    fig.savefig(OUT/'pure_e.pdf')
    plt.close(fig)

    summary = ROOT/'residual_map_upsample_summary.npz'
    if summary.exists():
        z = np.load(summary)
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, layout='constrained')
        sel = (S.ELL >= 3500) & (S.ELL < 4800)
        for ax, proj in zip(axes.flat, S.PROJS):
            key = f'{proj}_BB'
            if key+'_pad0_native' not in z:
                continue
            model = z[key+'_model'][sel]
            floor = proj == 'ProjSTG'
            for pad, factor, style, color, label in [
                (0, 'native', '-', PS.PROJ_COLORS[proj], 'native'),
                (0, 'fine', '--', '.3', 'twice-finer'),
                (128, 'fine', ':', '.1', 'twice-finer, padded')]:
                value = z[f'{key}_pad{pad}_{factor}'][:, sel]/model-(0 if floor else 1)
                y, e = value.mean(0), value.std(0, ddof=1)/np.sqrt(len(value))
                scale = 1e6 if floor else 100
                ax.plot(S.ELL[sel], scale*y, style, color=color, label=label)
                ax.fill_between(S.ELL[sel], scale*(y-e), scale*(y+e), color=color, alpha=.1)
            ax.text(.04, .94, PS.PROJ_LABEL[proj], transform=ax.transAxes, va='top')
            ax.set_ylabel(r'$10^6 BB/EE_{\rm model}$' if floor else 'Residual [%]')
            ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8, loc='lower left')
        for ax in axes[-1]:
            ax.set_xlabel(r'$\ell$')
        fig.savefig(OUT/'production_bb.png', dpi=160)
        fig.savefig(OUT/'production_bb.pdf')
        plt.close(fig)
    plot_ring_controls()


if __name__ == '__main__':
    main()
