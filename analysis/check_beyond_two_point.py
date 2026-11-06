"""Geometry, smoothing, peak and raw-moment gates for lognormal controls."""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/stages'))
from beyond_two_point import chart_grid, radius, statistics, SIGMA

for chart in ('TAN', 'STG', 'ZEA'):
    t, p, mask, w, g, mu = chart_grid(chart, 30, .1)
    c = np.arccos(np.sin(t)*np.cos(p))
    np.testing.assert_allclose(c[mask].max(), np.deg2rad(15), atol=.002)
    # All charts cover one spherical cap, regardless of projected area.
    np.testing.assert_allclose(w.sum()*np.deg2rad(.1)**2,
                               2*np.pi*(1-np.cos(np.deg2rad(15))), rtol=.002)
    if chart == 'STG':
        assert np.max(abs(g)) == 0
    if chart == 'ZEA':
        np.testing.assert_array_equal(mu, 1)
    # A scalar evaluated without filtering is identical under pullback.
    raw = np.sin(3*t)*np.cos(2*p)
    s = gaussian_filter(raw, SIGMA/np.deg2rad(.1))
    v = statistics(raw, s, s*0, mask, w, .5)
    z = statistics(raw, raw, raw, mask, w, .5)
    np.testing.assert_array_equal(v[:2], z[:2])
# Point extrema survive any smooth invertible coordinate change; this
# finite-pixel test also fixes the sign and threshold of our counter.
y, x = np.mgrid[-20:21, -20:21]
source = np.exp(-(x*x+y*y)/20)
mask = np.ones_like(source, dtype=bool)
out = statistics(source, source, source*0, mask, np.ones(mask.sum()), .5)
assert out[-1] == 1
if '--results' in sys.argv:
    from beyond_two_point_report import summarize
    summary = summarize()
    # Conformal-chart third-moment response cannot be a shear-only law.
    stg = next(r for r in summary['records']
               if r['chart'] == 'STG' and r['diameter'] == 60.)
    third = stg['statistics']['smooth_third']
    assert stg['g2'] == 0 and third['bias'] > 5*third['se']
    for chart in ('TAN', 'STG'):
        stat = next(r for r in summary['records']
                    if r['chart'] == chart and r['diameter'] == 60.)['statistics']['smooth_third']
        assert .8 < stat['bootstrap_se']/stat['se'] < 1.2
    # These are sampling-control tolerances, not physical upper limits.
    for key, tolerance in [('smooth_variance', .001), ('smooth_third', .003),
                           ('wavelet_l1', .001), ('peaks_gt_2sigma', .03)]:
        drift = max(abs(r['statistics'][key]['controls']['both']['paired_drift'])
                    for r in summary['records'])
        assert drift < tolerance, (key, drift)
    print('24 paired skies and 12 resolution controls pass')
print('beyond-two-point geometry and estimator gates pass')
