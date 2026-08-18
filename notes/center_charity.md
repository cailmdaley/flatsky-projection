# Is any projection given an uncharitable centre?

**Verdict: no.** All five projections are built with the same centre
`(alpha0, delta0) = (0, -56)`, which is also the centre of every analysis
mask, and that common centre is the *exact* minimum of the mask-weighted
shear for every projection that responds to the centre at all. Nothing is
penalized by centre placement, and no recentering is available that would
help.

The distortion visible in the figures is SFL's, and it is irreducible: SFL's
shear does not depend on `delta_center` at any order. See below.

## 1. Where each projection's origin actually is

`FlatSkyProjection` splits into two families (`maps/src/FlatSkyProjection.cxx`).

**Zenithal (SIN=Proj2, STG=Proj4, ZEA=Proj5) — genuinely tangent at
`(alpha0, delta0)`.** `QuatToXY` rotates the sky into the projection frame
before applying any radial formula:

```cxx
// FlatSkyProjection.cxx:388
auto qr = ~q0_ * q * q0_;
double cc = qr.b();
...
case Proj5: { k = rad * sqrt(2. / (1. + cc)); break; }   // :405
double x = k * qr.c();
double y = -k * qr.d();
```

and the rotator is built from both centre angles:

```cxx
// FlatSkyProjection.cxx:196, :206
q0_ = get_origin_rotator(alpha0_, delta0_);
```

So `cc` is the cosine of the angular distance **from `(alpha0, delta0)`**.
ZEA is tangent at the field centre, *not* at the pole. Same for SIN and STG.
This is the crux question and the answer is the charitable one.

**Cylindrical (SFL=Proj0, BICEP=Proj9) — `cyl_ = true` (:183), no rotator
used.** Their formulae are written directly in `(alpha, delta)`:

```cxx
// FlatSkyProjection.cxx AngleToXY
case Proj0: { x = dalpha * COS(delta / rad); y = delta0_ - delta; break; }   // :347
case Proj9: { x = dalpha * cosdelta0_;       y = delta0_ - delta; break; }   // :362
```

- **SFL**: `delta0_` appears *only* as an additive offset in `y`. The
  Jacobian in the orthonormal (east, north) sky basis is
  `diag(1, 1)` corrected by `d/d(east) of dalpha*cos(delta)` — a function of
  `delta` alone. **SFL's shear is exactly independent of `delta_center`.**
  `alpha0_` does matter (it sets where `dalpha = 0`), and `alpha0 = 0` is
  already the mask centre.
- **BICEP**: `cosdelta0_` is a real scale factor, so `delta0` does change the
  geometry: `J = diag(cos(delta0)/cos(delta), 1)`, shear zero exactly on the
  `delta = delta0` line. `delta0 = -56` puts that line through the mask
  centre — again the charitable choice.

**Pipeline centres, confirmed:** `astra.yaml:171,204,231,267` and
`grid/overnight.sh:58,68` all pass `--ra-deg 0 --dec-deg -56`;
`_project.project_frame` forwards `delta_center=dec_deg*deg` and deliberately
does not pass `alpha_center`, because `FlatSkyMap`'s default is already 0
(`_project.ALPHA_CENTER_DEG_DEFAULT`, with the verification note). The masks
are disks about the same `(0, -56)` (`astra.yaml:171-172`,
`make_masks.py:179`). And the paper's closed-form machinery uses the identical
values — `geometry.ALPHA_CENTER_DEG = 0.0`, `geometry.DELTA_CENTER_DEG = -56.0`.
**No consistency bug: the g-table and the pipeline agree.**

## 2. Centre sensitivity, closed form

`analysis/center_charity.py`. mask^2-weighted (the pseudo-C_l weighting) over
the 2021 C2/2.5deg apodized disks about (0, -56), Nside 512.

`<g^2>`, `g = (s1-s2)/(s1+s2)`, largest disk (0.65 rad = 37.2 deg):

| dec_center | SFL | SIN | STG | ZEA | BICEP |
|---|---|---|---|---|---|
| -90 | 6.312e-02 | 3.810e-02 | 2.6e-10 | 7.232e-03 | 1.000e+00 |
| -70 | 6.312e-02 | 7.228e-03 | 2.6e-10 | 1.590e-03 | 1.071e-01 |
| -61 | 6.312e-02 | 3.872e-03 | 2.6e-10 | 8.896e-04 | 6.154e-02 |
| **-56** | **6.312e-02** | **3.429e-03** | 2.6e-10 | **7.948e-04** | **5.632e-02** |
| -51 | 6.312e-02 | 3.872e-03 | 2.6e-10 | 8.896e-04 | 5.830e-02 |
| -36 | 6.312e-02 | 1.201e-02 | 2.6e-10 | 2.545e-03 | 8.133e-02 |
| 0 | 6.312e-02 | 1.933e-01 | 2.6e-10 | 2.837e-02 | 1.194e-01 |

Smallest disk (0.175 rad = 10 deg) is far sharper — SIN/ZEA fall by ~3 orders
of magnitude from dec_center 0 to -56, and SFL is again exactly flat.

A 1-degree scan over dec_center in [-64, -46] for radii 0.65 / 0.45 / 0.175
puts the argmin at -56 in eight of nine cases; the ninth (BICEP, 0.65 rad) is
-55 with `<g^2>` lower by 0.1%. **-56 is the optimum, to within noise, for
every projection and every mask.**

RA scan (dec_center = -56, largest disk): `<g^2>` at alpha_center = 0 / 30 / 90
is 6.31e-02 / 8.81e-02 / 2.55e-01 for SFL and 3.43e-03 / 9.02e-03 / 9.89e-02
for SIN. Zero is the minimum; BICEP is alpha_center-independent as its
formula requires. Good.

## 3. Magnification

`<(mu-1)^2>`, `mu = |det J|`, largest disk: SFL is 1e-25 (equal-area,
roundoff), ZEA 1e-9 (equal-area), STG 1.48e-02 and SIN 1.18e-02 both minimized
at -56. BICEP reads 7.3 at -56 and *falls* toward -90 — but that is a
degenerate collapse, not an improvement: BICEP's `mu = cos(delta0)/cos(delta)`
goes to 0 identically as `delta0 -> -90`, so `(mu-1)^2 -> 1` by everything
vanishing. The genuine statement is that the 37-degree disk reaches the south
pole, where BICEP's cylindrical scaling is singular, and no centre fixes that.
It is a property of the projection, which is what the paper is measuring.

## 3b. The pole, and joint recentering

The 0.65 rad disk about dec -56 runs from dec -18.8 to **-89.9** — it contains
the south celestial pole, which is why SFL smears into horizontal streaks and
BICEP's bottom becomes a flat full-width band in `figures/masks_projections.png`.
The zenithal three are untroubled: the pole is an ordinary point 34 deg from
their tangent point.

But the pole is not a *discrete* pathology worth excising. Splitting the
mask^2 weight (largest disk, everything at (0,-56)):

| region | % of mask^2 weight | SFL | SIN | ZEA | BICEP |
|---|---|---|---|---|---|
| dec < -85 | 1.4% | 5.7% of `<g^2>` | 3.2% | 3.1% | 17.2% |
| dec < -80 | 4.7% | 14.6% | 8.5% | 8.5% | 39.6% |

BICEP is the one that concentrates — 40% of its shear from the outer 5% of the
mask — but SFL's 63% still comes from dec > -80. SFL's streaking is broad
meridian convergence, not a pole singularity. (STG's row is omitted: its
`<g^2>` is 2.6e-10, pure roundoff, so the "99.99% from the pole" the script
prints is meaningless.)

**Joint recentering** — moving *both* the mask disk and the projection centre
to a common dec, so the R=0.65 disk clears the pole:

| dec_center | disk south edge | SFL | SIN | STG | ZEA | BICEP |
|---|---|---|---|---|---|---|
| **-56** | -93.2 | 6.312e-02 | 3.429e-03 | ~0 | 7.948e-04 | 5.632e-02 |
| -53 | -90.2 | 4.878e-02 | 3.429e-03 | ~0 | 7.948e-04 | 4.739e-02 |
| -50 | -87.2 | 3.885e-02 | 3.429e-03 | ~0 | 7.948e-04 | 3.904e-02 |
| -47 | -84.2 | 3.138e-02 | 3.429e-03 | ~0 | 7.948e-04 | 3.203e-02 |
| -43 | -80.2 | 2.386e-02 | 3.429e-03 | ~0 | 7.948e-04 | 2.457e-02 |

Two things fall out.

1. **SIN, STG and ZEA are exactly invariant** — 5 significant figures, not
   approximately. They must be: the disk sits *on* the tangent point, so
   rotating the whole configuration is an isometry. Their `<g^2>` is a pure
   function of disk radius. This confirms the tangent-point result of Sec. 1
   the hard way, and it means the zenithal projections have no centre choice
   to be charitable or uncharitable about.
2. **SFL and BICEP fall monotonically toward the equator** — 38% lower at
   dec -50, 62% lower at -43, with no kink at the pole crossing (-53, edge
   -90.2, is already 23% below -56). The same slope appears for R = 0.45,
   which never touches the pole at all. So there is no sweet spot to find and
   no pathology to dodge: their shear is simply a monotone function of the
   field's declination, and the only "optimum" is dec 0, which is not
   anybody's field.

## 4. Recommendation

Keep the current configuration and *state it as a policy* in the paper: **a
single common centre at the SPT field centre (RA 0, dec -56), shared by all
five projections and by every mask, fixed once from the instrument's footprint
and never tuned per projection.** That is what the code already does.

It is the right policy for three reasons, in increasing order of force.

1. **Nothing is being handicapped relative to its own best case.** Holding the
   masks fixed, dec -56 is the exact `<g^2>` minimum over `delta_center` for
   SIN, ZEA and BICEP (Sec. 2), and alpha_center 0 is the minimum for SFL,
   SIN and ZEA. No projection would prefer a different centre.
2. **Three of the five have no centre freedom at all.** Under a joint
   recentering of masks and projections, the zenithal three are exactly
   invariant (Sec. 3b). So "optimizing the centre" is not a knob that trades
   projections off against each other — it is a knob that moves SFL and BICEP
   and leaves SIN/STG/ZEA untouched.
3. **Therefore the only alternative to a physically anchored centre is a
   tuned one.** SFL's and BICEP's shear decreases monotonically all the way to
   the equator with no natural stopping point. Any dec other than -56 would be
   a number chosen to make two projections look better, with no principle
   behind it. Anchoring on the real field centre is the defensible choice
   precisely because it is not free.

**The law is unaffected either way.** The forward model and the measurement
share the same Jacobian, so `<g^2>` moving with the centre moves prediction
and data together — the `BB/EE ~ 2<g^2>` relation is centre-independent. What
the centre changes is only the *spread between projections* in the comparison
figures, i.e. presentation, not physics. Worth saying explicitly in the paper.

On the figure that prompted this: the badly distorted mask is SFL's largest
disk, and the distortion is meridian convergence approaching the south pole
(the disk's southern edge is at dec -93.2, so it contains the pole). Two
sentences are worth adding to the paper, because a reader will have the same
instinct Cail did: (i) *translating* the tangent point does not help — with the
masks held fixed, SFL's `<g^2>` is constant to all printed digits across the
full dec_center range, because `delta0` enters SFL only as an additive offset
in `y`; and (ii) moving mask and projection together does reduce it, but only
by walking toward the equator, which would be tuning rather than a choice.
