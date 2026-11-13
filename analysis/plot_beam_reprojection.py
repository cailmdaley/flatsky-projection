"""A 1 arcmin Gaussian on the SFL grid against the same Gaussian on the sphere.

Reads the `beam_response` output (measured responses, reprojected to ZEA) and
the `beam_summary` output (its local model, target_prediction.npz).

  left   SFL/sphere response - 1 against ell on the 5% disk and the 1500d
         field: points measured, lines the model
  right  on the 1500d field at the bin nearest ell = 3000, the response of
         each filter against mode direction chi, measured and modelled

Run:  bash env.sh python3 analysis/plot_beam_reprojection.py
          --beam-response <dir> --beam-summary <dir> --out <dir>
Out:  <out>/beam_reprojection.pdf, which the paper includes, and
      <out>/fig_beam_reprojection.png
"""

import argparse
from pathlib import Path

import numpy as np

import paper_style as PS
import spectra as S


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "beam_response", "beam_summary")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.beam_response/'summary.npz') as f:
        data = {k: f[k] for k in f.files}
    with np.load(args.beam_summary/'target_prediction.npz') as f:
        model = {k: f[k] for k in f.files}
    np.testing.assert_allclose(model['ell'], data['ell'])
    np.testing.assert_allclose(model['chi'], data['chi'])

    with plt.rc_context(PS.rc()):
        fig, axs = plt.subplots(1, 2, figsize=(PS.TEXT_W, 2.4), layout='constrained')
        ell = data['ell']
        index = np.argmin(abs(ell-3000))
        ax = axs[0]
        for footprint, label, color in (('disk5', r'5\% disk', PS.SECONDARY[1]),
                                        ('field1500d', '1500d field', PS.PROJ_COLORS['ProjSFL'])):
            prefix = footprint+'_ProjZEA_'
            ax.plot(ell, 100*(data[prefix+'sfl_response']/data[prefix+'sphere_response']-1),
                    'o', color=color, ms=2.4, label=label)
            ax.plot(ell, 100*(model[footprint+'_source']/model[footprint+'_sphere']-1), color=color)
        ax.set(xlim=(500, 4000), xlabel=r'$\ell$', ylabel=r'$R_{\rm SFL}/R_{\rm sphere}-1$ [\%]')
        ax.legend()
        ax = axs[1]
        footprint = 'field1500d'
        prefix = footprint+'_ProjZEA_'
        chi = np.rad2deg(data['chi'])
        for tag, key, label, color, marker in (
                ('sfl', '_source2d', 'Filtered on SFL', PS.PROJ_COLORS['ProjSFL'], 'o'),
                ('sphere', '_sphere2d', 'Filtered on the sphere', PS.SECONDARY[1], '+')):
            ax.plot(chi, data[prefix+tag+'_response2d'][index], marker, color=color, ms=3, label=label)
            ax.plot(chi, model[footprint+key][index], color=color)
        ax.set(xlim=(0, 180), xlabel=r'Mode direction $\chi$ [deg]', ylabel='Power response')
        ax.legend()
        args.out.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out/'beam_reprojection.pdf')
        fig.savefig(args.out/'fig_beam_reprojection.png', dpi=180)
        plt.close(fig)
    print(args.out/'beam_reprojection.pdf')


if __name__ == '__main__':
    main()
