"""Run under an spt3g build with flattenpol-polar-angle.diff applied: flatten_pol must equal the python polar rotation."""
import numpy as np
from spt3g import core, maps
import spt3g, sys
print("spt3g from", spt3g.__file__)

R = "/home/cailmdaley/work/pol_project/results/highell"
wrap = lambda v: (v + np.pi) % (2 * np.pi) - np.pi
for proj in ("ProjSFL", "ProjSIN", "ProjZEA", "ProjSTG", "ProjBICEP"):
    fr = next(core.G3File(f"{R}/projected_qu/qu_{proj}_seed0000.g3"))
    Qs = fr["Q"]
    ny, nx = Qs.shape
    lo, hi = 900, 2400  # central block, keeps it quick
    yy, xx = np.mgrid[lo:hi, lo:hi].astype(float)
    x, y = xx.ravel(), yy.ravel()
    h = 1e-3
    def ang(a, b):
        aa, dd = Qs.xy_to_angle(core.DoubleVector(a), core.DoubleVector(b))
        return np.asarray(aa), np.asarray(dd)
    a1, d1 = ang(x - h, y); a2, d2 = ang(x + h, y)
    a3, d3 = ang(x, y - h); a4, d4 = ang(x, y + h)
    dax, ddx = wrap(a2 - a1) / (2 * h), (d2 - d1) / (2 * h)
    day, ddy = wrap(a4 - a3) / (2 * h), (d4 - d3) / (2 * h)
    _, dc = ang(x, y); cd = np.cos(dc)
    rot_s = (np.arctan2(-dax, day) + np.arctan2(-ddy, -ddx)).reshape(xx.shape)
    rot_p = (2 * np.arctan2(-ddx - cd * day, ddy - cd * dax)).reshape(xx.shape)
    s = -1.0  # COSMO
    zs = np.asarray(fr["Q"])[lo:hi, lo:hi] + 1j * np.asarray(fr["U"])[lo:hi, lo:hi]
    zu = zs * np.exp(-1j * s * rot_s)          # unflatten with the stock angle
    zp_py = zu * np.exp(1j * s * rot_p)        # python polar flattening

    # Fresh unflattened maps on the same grid, flattened by the patched C++.
    qa = np.zeros((ny, nx)); ua = np.zeros((ny, nx))
    qa[lo:hi, lo:hi] = np.nan_to_num(zu.real); ua[lo:hi, lo:hi] = np.nan_to_num(zu.imag)
    q = maps.FlatSkyMap(qa, Qs.res, proj=Qs.proj, alpha_center=Qs.alpha_center,
                        delta_center=Qs.delta_center, pol_conv=maps.MapPolConv.COSMO,
                        pol_type=maps.MapPolType.Q, flat_pol=False)
    u = maps.FlatSkyMap(ua, Qs.res, proj=Qs.proj, alpha_center=Qs.alpha_center,
                        delta_center=Qs.delta_center, pol_conv=maps.MapPolConv.COSMO,
                        pol_type=maps.MapPolType.U, flat_pol=False)
    maps.flatten_pol(q, u)
    zc = np.asarray(q)[lo:hi, lo:hi] + 1j * np.asarray(u)[lo:hi, lo:hi]
    ok = np.isfinite(zp_py) & (np.abs(zp_py) > 0)
    err = np.abs(zc - zp_py)[ok].max() / np.abs(zp_py[ok]).max()
    err_s = np.abs(zc - zs)[ok].max() / np.abs(zs[ok]).max()
    dmax = np.degrees(np.abs(wrap(rot_s - rot_p)[ok.reshape(xx.shape)]).max() / 2)
    print(f"{proj}: patched flatten_pol vs python polar: max rel err {err:.1e}; "
          f"vs stock-flattened map {err_s:.1e}; max |Delta| in block {dmax:.3f} deg")
