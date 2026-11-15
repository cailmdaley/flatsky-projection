# projcoupling

A flat-sky power spectrum measured on a projected map is not the sky's. The
projection's local Jacobian stretches each mode, so a band in plane wavenumber
reads the sky spectrum over a range of multipoles, and its shear turns modes, so
E leaks into B. `projcoupling` builds the matrix that maps the sky's spectra to
the bandpowers measured on the grid,

    C^X_b = Σ_l' R^{X←Y}_{b l'} C^Y_l'
    R^{X←Y}_{b l'} = (1/N) ∫ d²n W² μ² ⟨K^{X←Y}(χ) δ(l' − k ρ(χ))⟩_{χ, k∈b},   N = ∫ d²n W² μ

for TT←TT, EE←EE, BB←BB, BB←EE, EE←BB and TE←TE, from nothing but the sky
coordinates of the grid's pixels and the analysis mask. NaMaster removes the
mask's coupling; R removes the projection's. The leading-order leakage is
BB/EE = 2⟨g²⟩, with g = (s1 − s2)/(s1 + s2) the projection's shear (`gsq`).

```python
import numpy as np, pymaster as nmt, projcoupling as pc

geo = pc.Geometry.from_wcs(wcs, mask.shape)       # or Geometry.from_radec(ra, dec, res_rad)
edges, ell_in = np.arange(500, 4001, 50), np.arange(8001)
R = pc.coupling_matrix(geo, mask, edges, ell_in)  # about a second for a 1500 deg² field at 2'

bins = nmt.NmtBinFlat(edges[:-1], edges[1:])
f = nmt.NmtFieldFlat(Lx, Ly, mask, [q, u], purify_b=True)
cl = nmt.compute_full_master_flat(f, f, bins)     # mask decoupled
sky = pc.decouple(pc.namaster_spectra(cl), R, {"EE": fid_ee, "BB": fid_bb}, ell_in, edges)
print(sky["BB"], 2 * pc.gsq(geo, mask))           # projection decoupled; leading-order leak
```

`decouple` fits one amplitude per sky band (pass coarser `sky_edges` for a
least-squares fit) times a fiducial shape, EE and BB jointly, and holds the
multipoles outside the sky bands at the fiducial, so the fiducial must be a
model of the sky in absolute units, not just a shape. `ell_in` must reach
`edges[-1] · max(s1)`.

**Conventions.** A plane mode of wavenumber k is the spherical multipole
l = k − ½ (`k_offset`). Bandpowers are uniform averages over k in [lo, hi).
Polarization is assumed flattened by the polar angle of the Jacobian, so only
its symmetric stretch (s1, s2) acts; a pipeline that flattens by another angle
leaks an extra O(g⁴). R depends on a pixel only through (s1, s2), so pixels are
histogrammed in (log s1, log s2) (`n_cells` per axis); the cost is set by the
cells, not the pixels.

**The one approximation is locality.** The projection's shear varies on the
scale of the field, multipoles L of a few, and R applies it locally to modes at
l ≫ L, which errs at order (L/l)² ~ 10⁻³ for l ≳ 1000.

Tests: `pytest projcoupling/tests` (needs astropy).
