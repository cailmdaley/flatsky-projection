"""The beam reprojection experiment against its local model, with its B controls.

Reads the `beam_response` output: a 1 arcmin Gaussian applied on the SFL grid
and on the sphere, each beside an unfiltered twin, reprojected to ZEA and STG,
with filtered/unfiltered power binned in multipole and mode direction on the
5% disk and the 1500d field. Writes three products under --out:

  target_prediction.npz  the local model of those responses
                         (`section5_geometry.beam_target_prediction`),
                         integrated over each measured (ell, chi) bin with
                         4 x 4 Gauss-Legendre points; keys
                         <footprint>_{source,sphere}[2d]
  b_diagnostics.npz      purified EE/BB of the unfiltered maps, their SFL -
                         sphere mismatch and the differential filtered map,
                         and the differential map's cross spectra, decoupled
                         with the production pure-B workspaces of
                         `coupling_matrices`; keys <footprint>_<proj>_<tag>
  beam_summary.json      the numbers Sect. 6.2 quotes: at the bin nearest
                         ell = 3000, the measured and predicted SFL/sphere
                         response difference on ZEA, its 2D sector range,
                         band means over 1000-2000, 2000-3000 and 3000-4000,
                         the largest sector residual against the model and
                         the slope of the measured cos 2chi amplitude on the
                         predicted one over 1000 <= ell <= 3000; and the B
                         diagnostics as band means over 1000 <= ell < 3000
                         relative to the sphere baseline's EE

Run:  bash env.sh python3 analysis/beam_summary.py
          --beam-response <dir> --coupling-matrices <dir> --flatsky-masks <dir>
          --healpix-masks <dir> --theory-spectra <npz> --out <dir>
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
# Load NaMaster's CFITSIO before spt3g (via geometry), which loads an older ABI.
import pymaster as nmt

import section5_geometry as G5
import spectra as S
import theory as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'stages'))
from namaster_spectra import load_mask  # noqa: E402

FOOTPRINTS = (('disk5', 'disk_5'), ('field1500d', S.FIELD))
PROJECTIONS = ('ProjZEA', 'ProjSTG')


def target_model(ell, chi, theory, healpix_masks):
    """Integrate numerator and denominator over each measured Fourier bin."""
    gx, gw = np.polynomial.legendre.leggauss(4)
    le = (ell[:, None] + (ell[1]-ell[0])/2*gx).ravel()
    ca = (chi[:, None] + (chi[1]-chi[0])/2*gx).ravel()
    raw = G5.beam_target_prediction(le, ca, theory, healpix_masks)
    out = {'ell': ell, 'chi': chi}
    for footprint, terms in raw.items():
        integrated = []
        for term in terms:
            z = term.reshape(len(ell), 4, len(chi), 4)
            weight = (gw[None, :, None, None]*gw[None, None, None, :]
                      * le.reshape(len(ell), 4)[:, :, None, None])
            integrated.append(np.sum(z*weight, axis=(1, 3)))
        ns, ds, nr, dr = integrated
        out[footprint+'_source2d'] = ns/ds
        out[footprint+'_sphere2d'] = nr/dr
        out[footprint+'_source'] = ns.sum(1)/ds.sum(1)
        out[footprint+'_sphere'] = nr.sum(1)/dr.sum(1)
    return out


def b_diagnostics(beam_response, coupling_matrices, flatsky_masks):
    """Decouple every map pair with the production pure-B workspaces, cross terms included."""
    out = {}
    for proj in PROJECTIONS:
        pairs = {tag: np.load(beam_response/f'{tag}_{proj}.npy') for tag in
                 ('sfl_base', 'sfl_beam', 'sphere_base', 'sphere_beam')}
        pairs['mismatch'] = pairs['sfl_base']-pairs['sphere_base']
        pairs['differential'] = ((pairs['sfl_beam']-pairs['sfl_base'])
                                 - (pairs['sphere_beam']-pairs['sphere_base']))
        for footprint, name in FOOTPRINTS:
            index = json.loads((coupling_matrices/f'index_{proj}_{name}.json').read_text())
            lo = np.arange(0., index['lmax'], index['delta_ell'])
            bins = nmt.NmtBinFlat(lo, lo+index['delta_ell'])
            out['ell'] = lo+index['delta_ell']/2
            mask, (ys, xs), lx, ly = load_mask(flatsky_masks, proj, name)
            workspace = nmt.NmtWorkspaceFlat()
            workspace.read_from(str(coupling_matrices/f'w22_{proj}_{name}.fits'))
            fields = {tag: nmt.NmtFieldFlat(lx, ly, mask, list(pair[:, ys, xs]),
                                            purify_b=index['purify_b'])
                      for tag, pair in pairs.items()}
            prefix = f'{footprint}_{proj}_'
            for tag, fld in fields.items():
                coupled = nmt.compute_coupled_cell_flat(fld, fld, bins)
                out[prefix+tag] = workspace.decouple_cell(coupled)
            for other in ('mismatch', 'sfl_base', 'sphere_base'):
                cl = nmt.compute_coupled_cell_flat(fields['differential'], fields[other], bins)
                out[prefix+'cross_'+other] = workspace.decouple_cell(cl)
            print('Decoupled B controls', footprint, proj, flush=True)
            del fields
        del pairs
    return out


def summary(data, model, bdiag):
    idx = np.argmin(abs(data['ell']-3000))
    result = {'ell_response': float(data['ell'][idx]), 'response': {}, 'b_diagnostic': {}}
    for footprint, _ in FOOTPRINTS:
        p = footprint+'_ProjZEA_'
        measured = data[p+'sfl_response'][idx]/data[p+'sphere_response'][idx]-1
        predicted = model[footprint+'_source'][idx]/model[footprint+'_sphere'][idx]-1
        result['response'][footprint] = {
            'fractional_difference': float(measured), 'prediction': float(predicted),
            'sfl_mean': float(data[p+'sfl_response'][idx]),
            'sphere_mean': float(data[p+'sphere_response'][idx]),
            'sfl_sector_min': float(data[p+'sfl_response2d'][idx].min()),
            'sfl_sector_max': float(data[p+'sfl_response2d'][idx].max())}
        response = result['response'][footprint]
        response['bands'] = {}
        measured_ratio = data[p+'sfl_response']/data[p+'sphere_response']-1
        predicted_ratio = model[footprint+'_source']/model[footprint+'_sphere']-1
        for lo, hi in ((1000, 2000), (2000, 3000), (3000, 4000)):
            band = (data['ell'] >= lo) & (data['ell'] < hi)
            response['bands'][f'{lo}_{hi}'] = {
                'measured': float(measured_ratio[band].mean()),
                'predicted': float(predicted_ratio[band].mean())}
        clean = (data['ell'] >= 1000) & (data['ell'] <= 3000)
        residual = data[p+'sfl_response2d']-model[footprint+'_source2d']
        response['sector_model_max_abs_1000_3000'] = float(abs(residual[clean]).max())
        sphere_residual = data[p+'sphere_response2d']-model[footprint+'_sphere2d']
        response['sphere_sector_model_max_abs_1000_3000'] = float(abs(sphere_residual[clean]).max())
        cosine = np.cos(2*data['chi'])
        measured2d = data[p+'sfl_response2d']/data[p+'sphere_response2d']-1
        predicted2d = model[footprint+'_source2d']/model[footprint+'_sphere2d']-1
        observed_amplitude = 2*np.mean(measured2d*cosine, axis=1)
        predicted_amplitude = 2*np.mean(predicted2d*cosine, axis=1)
        response['quadrupole_amplitude_slope_1000_3000'] = float(
            np.dot(observed_amplitude[clean], predicted_amplitude[clean])
            / np.dot(predicted_amplitude[clean], predicted_amplitude[clean]))
        keep = (bdiag['ell'] >= 1000) & (bdiag['ell'] < 3000)
        for proj in PROJECTIONS:
            p = f'{footprint}_{proj}_'
            ee = bdiag[p+'sphere_base'][0, keep].mean()
            vals = {tag: float(bdiag[p+tag][3, keep].mean()/ee) for tag in
                    ('sfl_base', 'sfl_beam', 'sphere_base', 'sphere_beam', 'mismatch', 'differential')}
            for other in ('mismatch', 'sfl_base', 'sphere_base'):
                vals['cross_'+other] = float(bdiag[p+'cross_'+other][3, keep].mean()/ee)
            vals['mismatch_power_change'] = vals['differential'] + 2*vals['cross_mismatch']
            result['b_diagnostic'][footprint+'_'+proj] = vals
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    S.add_inputs(ap, "beam_response", "coupling_matrices", "flatsky_masks",
                 "healpix_masks", "theory_spectra")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with np.load(args.beam_response/'summary.npz') as f:
        data = {k: f[k] for k in f.files}
    model = target_model(data['ell'], data['chi'], T.load(args.theory_spectra),
                         args.healpix_masks)
    np.savez(args.out/'target_prediction.npz', **model)
    bdiag = b_diagnostics(args.beam_response, args.coupling_matrices, args.flatsky_masks)
    np.savez(args.out/'b_diagnostics.npz', **bdiag)
    result = summary(data, model, bdiag)
    (args.out/'beam_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
