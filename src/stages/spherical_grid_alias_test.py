"""Paired out-of-band injection into the spherical-grid tests.

The white zero-B input cutoff is swept while reconstructing ell<=256
on Nside=128 and 384x768 CAR. The matched reference cuts the same alms to
256 *before* synthesis. Their difference isolates out-of-band response: sampling aliasing for point
samples, plus finite-pixel mode coupling for the averaged maps.
"""
import argparse
import json
from pathlib import Path
from spherical_grid_test import hp, np, synth, analyse, hp_inverse, hp_average, car_average, statistics
from _decisions import parse_apodization

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--input-lmax', type=int, required=True)
p.add_argument('--apodization', type=parse_apodization, required=True)
p.add_argument('--out', required=True)
args = p.parse_args()
np.random.seed(args.seed)
cl = np.ones(args.input_lmax+1)
cl[:2] = 0
sky = np.array([hp.synalm(cl, lmax=args.input_lmax, new=True), np.zeros(hp.Alm.getsize(args.input_lmax), complex)])
ell, mm = hp.Alm.getlm(256)
truth = sky[:, hp.Alm.getidx(args.input_lmax, ell, mm)]
records = {}
for label, a in [('inband', truth), ('extended', sky)]:
    lm = hp.Alm.getlmax(a.shape[-1])
    records[f'healpix_point_{label}'] = hp_inverse(np.asarray(hp.alm2map_spin(a, 128, 2, lm)), 256)
    records[f'car_point_{label}'] = analyse(synth(a, 384), 256)
    m, raw = hp_average(a, 128, 4)
    records[f'healpix_average_{label}'] = hp_inverse(m, 256)
    del m, raw
    m, raw = car_average(a, 384, 4)
    records[f'car_average_{label}'] = analyse(m, 256)
    del m, raw
for method in ('healpix_point', 'car_point', 'healpix_average', 'car_average'):
    records[f'{method}_alias'] = records[f'{method}_extended']-records[f'{method}_inband']
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
prefix = f'{args.input_lmax}-{args.seed}'
np.savez(out/f'{prefix}-alms.npz', sky=sky, truth=truth, **records)
result = dict(settings=dict(seed=args.seed, input_lmax=args.input_lmax, analysis_lmax=256, nside=128, factor=4),
              statistics={name: statistics(a, truth, 256, args.apodization) for name, a in records.items()})
(out/f'{prefix}-statistics.json').write_text(json.dumps(result, indent=2)+'\n')
