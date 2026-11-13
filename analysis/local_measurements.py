"""Local matched-filter experiment with exact spherical radial source injection.

Run with env.sh; no HEALPix sampling is applied to compact source profiles.
The CMB+noise calculation uses the local affine covariance, not full-sky CMB
realizations. Amplitudes are central surface brightness, not integrated flux.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import fft, special, integrate, optimize
from spt3g import core
import geometry as G

ARCMIN = np.pi / 10800


def profiles(r, kind, beam=1.2):
    sigma = beam / np.sqrt(8*np.log(2))
    if kind == 'point':
        return np.exp(-.5*(r/sigma)**2)
    target, conv = cluster_profile(beam)
    return np.interp(r, target, conv/conv[0], right=0.)


@lru_cache()
def cluster_profile(beam):
    """Beam-convolved, tapered beta-model SZ proxy; central brightness is unity."""
    sigma = beam / np.sqrt(8*np.log(2))
    axis = np.linspace(0, 20, 2001)
    source = (1+(axis/1.)**2)**(-1)*np.exp(-.5*(axis/8.)**2)
    target = np.linspace(0, 30, 1501)
    z = target[:, None]*axis[None, :]/sigma**2
    kernel = special.i0e(z)*np.exp(-.5*((target[:, None]-axis)/sigma)**2)
    conv = integrate.simpson(kernel*source*axis/sigma**2, x=axis, axis=1)
    return target, conv


def cutout(proj, ra, dec, n=256, res=.2):
    """Exact inverse chart around the source, with its centre at a pixel."""
    pixel = res*ARCMIN
    fsm = G._flat_map(proj, res_arcmin=res)
    xc, yc = fsm.angle_to_xy(ra*G.DEG, dec*G.DEG)
    v = (np.arange(n)-n//2)*pixel
    y, x = np.meshgrid(v, v, indexing='ij')
    rr, dd = fsm.xy_to_angle(core.DoubleVector((xc-x/pixel).ravel()),
                             core.DoubleVector((yc+y/pixel).ravel()))
    rr, dd = np.asarray(rr).reshape(x.shape), np.asarray(dd).reshape(x.shape)
    a, d = np.deg2rad([ra, dec])
    # Haversine avoids cancellation at sub-arcminute separation.
    radius = 2*np.arcsin(np.sqrt(np.clip(np.sin((dd-d)/2)**2+
                         np.cos(dd)*np.cos(d)*np.sin((rr-a)/2)**2, 0, 1)))/ARCMIN
    return x, y, radius


def fit(source, template, power, pixel):
    """2-D Fourier matched filter, with a continuous position maximum."""
    sf, tf = fft.fft2(source), fft.fft2(template)
    k = 2*np.pi*fft.fftfreq(source.shape[0], pixel)
    ky, kx = np.meshgrid(k, k, indexing='ij')
    w = np.conj(tf)/power
    norm = np.sum(abs(tf)**2/power).real
    def amplitude(pos):
        return (np.sum(w*sf*np.exp(1j*(kx*pos[0]+ky*pos[1]))).real/norm)
    # Positions in arcminutes for a well-scaled optimizer.
    result = optimize.minimize(lambda p: -amplitude(p*ARCMIN), [0., 0.],
                               method='BFGS', options={'gtol': 1e-10})
    return amplitude(np.zeros(2)), -result.fun, result.x, norm


def noisy_test(source, template, power, actual, pixel, nsim=64):
    """Independent pixel-white draws Fourier-coloured to the projected covariance.

    The compact sources are exact spherical samples; the backgrounds here are
    periodic local-affine Gaussian approximations to the projected CMB+noise.
    Fit maxima near a known 500 uK source; this is not a catalogue selection test.
    """
    rng = np.random.default_rng(4431)
    amplitudes, peaks, positions = [], [], []
    for i in range(nsim):
        noise = fft.ifft2(fft.fft2(rng.normal(size=source.shape))*np.sqrt(actual)/pixel).real
        image = 500*source+noise
        amp, peak, pos, _ = fit(image, template, power, pixel)
        amplitudes.append(amp); peaks.append(peak); positions.append(pos)
    return dict(nsim=nsim, input_uK=500, fixed_mean_uK=float(np.mean(amplitudes)),
                fixed_std_uK=float(np.std(amplitudes,ddof=1)),
                peak_mean_uK=float(np.mean(peaks)),
                position_mean_arcmin=np.mean(positions,axis=0).tolist(),
                position_std_arcmin=np.std(positions,axis=0,ddof=1).tolist())


def run(out, theory, res=.2, n=256, nsim=64):
    theory = np.load(theory)
    pixel = res*ARCMIN
    k = 2*np.pi*fft.fftfreq(n, pixel)
    ky, kx = np.meshgrid(k, k, indexing='ij')
    ell = np.hypot(kx, ky)
    beam = np.exp(-.5*(ell*(1.2/np.sqrt(8*np.log(2)))*ARCMIN)**2)
    cmb = np.interp(ell, theory['ell'], theory['lensed_TT'], right=0.)*beam**2
    white = (18*ARCMIN)**2
    rows = []
    for proj in ('ProjSFL', 'ProjZEA', 'ProjBICEP'):
        for ra in (-45., -22.5, 0., 22.5, 45.):
            for dec in (-68., -56., -44.):
                x, y, radius = cutout(proj, ra, dec, n, res)
                jac = G.jacobian(proj, [ra], [dec])[0]
                s1, s2 = G.singular_values(jac)
                mu = float(s1*s2)
                g = float((s1-s2)/(s1+s2))
                qx = jac[0,0]*kx+jac[1,0]*ky
                qy = jac[0,1]*kx+jac[1,1]*ky
                q = np.hypot(qx, qy)
                actual_cmb = mu*np.interp(q, theory['ell'], theory['lensed_TT'], right=0.)
                actual_cmb *= np.exp(-(q*(1.2/np.sqrt(8*np.log(2)))*ARCMIN)**2)
                for kind in ('point', 'cluster'):
                    source = profiles(radius, kind)
                    template = profiles(np.hypot(x,y)/ARCMIN, kind)
                    for noise, power in (('white', np.full_like(ell,white)),
                                          ('cmb_white', cmb+white)):
                        power = power.copy()
                        # Discard DC and large-scale backgrounds like a catalogue filter.
                        power[ell<300] = np.inf if noise=='cmb_white' else power[ell<300]
                        amp, peak, pos, norm = fit(source, template, power, pixel)
                        tf = fft.fft2(template)
                        # Exact linear-estimator variance with local projected sky covariance.
                        actual = actual_cmb+mu*white if noise=='cmb_white' else np.full_like(ell,mu*white)
                        variance = n*n*np.sum(abs(tf)**2*actual/power**2).real/(pixel**2*norm**2)
                        # The known local metric supplies an elliptical template.
                        tangent = np.linalg.solve(jac, np.stack([x.ravel(), y.ravel()]))
                        corrected = profiles(np.linalg.norm(tangent,axis=0).reshape(x.shape)/ARCMIN, kind)
                        corrected_amp = fit(source, corrected, power, pixel)[0]
                        magnified = profiles(np.hypot(x,y)/ARCMIN/np.sqrt(mu), kind)
                        magnified_amp = fit(source, magnified, power, pixel)[0]
                        monte_carlo = None
                        if nsim and ra == 45. and dec == -68. and noise == 'cmb_white':
                            monte_carlo = noisy_test(source, template, power, actual, pixel, nsim)
                        prediction = 2*mu/np.sqrt((1+s1*s1)*(1+s2*s2))
                        row = dict(projection=proj, ra=ra, dec=dec, profile=kind,
                                   noise=noise, mu=mu, g=g, amplitude=float(amp),
                                   peak_amplitude=float(peak), position_arcmin=pos.tolist(),
                                   sky_position_arcmin=np.linalg.solve(jac,pos).tolist(),
                                   magnification_template_amplitude=float(magnified_amp),
                                   sigma_uK=float(np.sqrt(variance)),
                                   metric_template_amplitude=float(corrected_amp),
                                   monte_carlo=monte_carlo,
                                   gaussian_prediction=float(prediction))
                        rows.append(row)
        print(proj, 'complete', flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(res_arcmin=res,n=n,beam_fwhm_arcmin=1.2,
                                  noise_uK_arcmin=18,rows=rows),indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--theory',type=Path,required=True,help='theory_spectra npz')
    p.add_argument('--res',type=float,default=.2)
    p.add_argument('--n',type=int,default=256)
    p.add_argument('--nsim',type=int,default=64)
    a=p.parse_args()
    run(a.out,a.theory,a.res,a.n,a.nsim)
