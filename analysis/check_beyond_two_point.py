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
    # A raw-coordinate check across independently sampled chart grids.
    assert max(r['raw_variance'] for r in summary['raw_cross_chart_spreads']) < .001
    assert max(r['raw_third'] for r in summary['raw_cross_chart_spreads']) < .005
    # The required interpretation gate: raw M3 alone is not a new
    # non-Gaussian response when S3=M3/M2^2 is consistent with zero.
    for row in summary['records']:
        s3 = row['statistics']['reduced_skewness']
        assert abs(s3['bias']) < 3*s3['se']
    # These are sampling-control regression tolerances, not physical limits.
    for key, tolerance in [('smooth_variance', .001), ('smooth_third', .003),
                           ('wavelet_l1', .001), ('peaks_gt_2sigma', .03)]:
        drift = max(abs(r['statistics'][key]['controls']['both']['paired_drift'])
                    for r in summary['records'])
        assert drift < tolerance, (key, drift)
    for key, tolerance in [('smooth_variance', .002), ('smooth_third', .005),
                           ('reduced_skewness', .005), ('wavelet_l1', .002),
                           ('peaks_gt_2sigma', .05)]:
        drift = max(abs(r['statistics'][key]['controls']['both']['paired_drift'])
                    for r in summary['calibrated_records'])
        assert drift < tolerance, (key, drift)
    tan = next(r for r in summary['records']
               if r['chart'] == 'TAN' and r['diameter'] == 60.)['statistics']
    matched = next(r for r in summary['calibrated_records']
                   if r['chart'] == 'TAN' and r['diameter'] == 60.)['statistics']
    assert abs(matched['smooth_third']['bias']) < .15*abs(tan['smooth_third']['bias'])
    assert abs(matched['peaks_gt_2sigma']['bias']) < .1*abs(tan['peaks_gt_2sigma']['bias'])
    # Calibration removes most moment/peak shifts, but NOT wavelet L1.
    wavelet = matched['wavelet_l1']
    assert wavelet['bias'] < -3*wavelet['se']
    assert abs(wavelet['bias']) > .5*abs(tan['wavelet_l1']['bias'])
    assert abs(wavelet['controls']['both']['paired_drift']) < .1*abs(wavelet['bias'])
    # Exercise the real writer and quoted-macro filter without editing paper.
    import tempfile
    from unittest.mock import patch
    import paper_numbers as PN
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root/'analysis').symlink_to(PN.ROOT/'analysis')
        (root/'paper').mkdir()
        (root/'paper/main.tex').write_text(
            r'\ngTANThirdLarge \ngTANReducedSkewnessLarge \ngCalTANThirdLarge')
        with patch.object(PN, 'ROOT', root):
            PN.beyond_two_point_numbers()
        text = (root/'paper/beyond_two_point_numbers.tex').read_text()
        assert r'\newcommand{\ngTANThirdLarge}{19.6\%}' in text
        assert r'\newcommand{\ngTANReducedSkewnessLarge}{0.57\%}' in text
        assert r'\newcommand{\ngCalTANThirdLarge}{0.9\%}' in text
        assert 'ngTANWaveletLarge' not in text
    print('24 paired skies, 24 scale-calibrated pairs and 16 resolution controls pass')
print('beyond-two-point geometry and estimator gates pass')
