# Cartographic lensing and lensing reconstruction

A derivation note on what a flat-sky map projection does to CMB lensing quadratic estimators and to four-point statistics generally, given the paper's result that the projection acts as a deterministic lens with shear $g = (s_1-s_2)/(s_1+s_2)$.

Everything here is closed-form or order-of-magnitude.
Numbers marked **[num]** come from `notes/curl.py` (finite-difference Jacobians of the five projections, reproduced in the appendix at the end); numbers marked **[unverified]** are estimates that need a real kernel calculation before they go anywhere near the paper.

**Headline results.**

1. The three zenithal projections (SIN, ZEA, STG) have *exactly* curl-free displacement — their effective deflection is the gradient of a potential I write down in closed form. SFL and BICEP do not: their grid axes track the local meridian, so their displacement carries a curl of order $0.25$ and $0.46$ rad rms over the SPT 1500d field. The gravitational curl (post-Born) is $\sim 10^{-5}$.
2. The cartographic deflection is *degrees*, 50–150× the gravitational $2.5'$ — but the cartographic convergence and shear are only $0.03$–$0.22$, i.e. comparable to gravitational lensing's $\kappa_{\rm rms}\approx 0.03$. The deflection is huge because the field is smooth, and the smoothness is what makes it harmless at high $L$. Distortion amplitude, not deflection amplitude, is the number that controls every bias below.
3. The dominant bias on $C_L^{\phi\phi}$ is not a lensing bias at all. It's the estimator normalization: the projection changes the observed $C_\ell$ by tens of percent, position-dependently, and the flat-sky QE assumes one isotropic spectrum. For SFL on the SPT 1500d field the local anisotropic response error is order unity.
4. The $EB$ estimator applied to a projected, unpurified map reconstructs the cartographic shear field itself. The leaked $B$ is therefore a template, not noise: subtracting it leaves only $2\langle g^2\rangle N_\ell^{EE}$, which for SFL on the 1500d field takes $BB/EE$ from $7\times10^{-2}$ — thirty times the lensed-$B$ floor — to $\sim 6\times10^{-4}$, below it.
5. Over the CMB ensemble at fixed projection the map stays exactly Gaussian. All connected $N$-point functions vanish. The entire signature sits in the off-diagonal two-point function, and it *masquerades* as a trispectrum only if you analyse assuming isotropy.

---

## 1. Effective lensing decomposition

### Setup, and what the fiducial has to be

The projection $P$ takes a sky direction to grid coordinates.
Its Jacobian $J$, defined between the local orthonormal sky frame $(\hat e_{\rm E}, \hat e_{\rm N})$ and the grid axes $(\hat x, \hat y)$, is dimensionless and unambiguous:
$$
J = \frac{\partial(x,y)}{\partial(E,N)}, \qquad dE = \cos\delta\, d\lambda, \quad dN = d\delta .
$$
$J = R\Sigma$ is the paper's polar decomposition; $\Sigma$ carries the shear that sets $BB/EE = 2\langle g^2\rangle$.

A *displacement field* needs more than $J$: it needs a flat reference chart, because the sphere has no canonical flat identity map.
The right choice is fixed by what the analysis actually does.
A flat-sky quadratic estimator FFTs the grid and treats the grid axes as globally fixed Cartesian axes.
So the fiducial is the tangent plane at the field centre in a globally fixed frame — the gnomonic chart TAN, with axes $(\xi,\eta)$ aligned to sky north at the tangent point.
Write $C = P_{\rm TAN}\circ P^{-1}$ for the composite map from the analysis grid to that plane, and
$$
\boxed{\;\vb*{d}(\vb*{x}) = C(\vb*{x}) - \vb*{x}\;}
$$
is the displacement: where the sky point at grid position $\vb*{x}$ really is, versus where the flat-sky analysis puts it.
$J_C = J_{\rm TAN} J^{-1}$, and $\psi_{ij} \equiv \partial_j d_i = (J_C)_{ij} - \delta_{ij}$.

The decomposition is standard:
$$
\kappa_{\rm eff} = -\tfrac12(\psi_{11}+\psi_{22}), \quad
\gamma_1^{\rm eff} = -\tfrac12(\psi_{11}-\psi_{22}), \quad
\gamma_2^{\rm eff} = -\tfrac12(\psi_{12}+\psi_{21}), \quad
\omega_{\rm eff} = \tfrac12(\psi_{12}-\psi_{21}),
$$
so that $\nabla\times\vb*{d} = 2\omega_{\rm eff}$ and $\vb*{d} = \grad\phi_{\rm cart} + \curl \Omega_{\rm cart}$ with $\omega_{\rm eff} = -\tfrac12\laplacian\Omega_{\rm cart}$.

Two caveats stated once, because both matter.
$\kappa_{\rm eff}$ and $\abs{\vb*{d}}$ are chart-dependent: TAN has its own $\kappa$ and its own $\abs{\gamma}$, and a different fiducial shifts both.
$\omega_{\rm eff}$ is *not* chart-dependent in the same way, because TAN with fixed axes is rotation-free (its Jacobian is symmetric in the polar frame, hence in any fixed frame), and so is any other point-anchored chart.
The shear that drives $E\to B$ — the paper's $g$, from $\Sigma$ alone — is chart-free and is the quantity to quote when the question is leakage.

### Which projections carry curl

**Zenithal projections are exactly curl-free.**
A zenithal projection sends angular radius $r$ from the tangent point to plane radius $R(r)$ at fixed azimuth.
In the orthonormal polar frames on both sides, $J = \mathrm{diag}(R'(r),\, R(r)/\sin r)$ — diagonal, hence symmetric, hence $\omega = 0$; and the composite against TAN, $J_C = \mathrm{diag}(\dd\tan r/\dd R,\, \tan r / R)$, is diagonal too.
The displacement is purely radial, $\vb*{d} = [\tan r(\rho) - \rho]\,\hat\rho$, and so is the gradient of
$$
\phi_{\rm cart}(\rho) = \int^{\rho}\!\big[\tan r(\rho') - \rho'\big]\dd\rho' ,
$$
which integrates in closed form for all three (taking the reference chart to be the radius-preserving ARC chart, $R=r$, gives the cleaner forms):
$$
\phi_{\rm SIN} = -\cos r - \tfrac{r^2}{2}, \qquad
\phi_{\rm ZEA} = -4\cos\tfrac r2 - \tfrac{r^2}{2}, \qquad
\phi_{\rm STG} = -4\ln\cos\tfrac r2 - \tfrac{r^2}{2}.
$$
Numerically $\abs{\omega} < 10^{-10}$ for all three at every offset tested, which is the finite-difference floor **[num]**.

**SFL and BICEP carry curl, because their grid axes track the local meridian.**
SFL's Jacobian in the sky orthonormal frame is the pure shear matrix
$$
J_{\rm SFL} = \begin{pmatrix} 1 & -t \\ 0 & 1\end{pmatrix}, \qquad t = (\lambda-\lambda_0)\sin\delta ,
$$
whose antisymmetric part is $-t/2$ — the same magnitude as its shear, $g_{\rm SFL}\simeq \abs{t}/2$.
So SFL's rotation and shear are *equal*: it is a pure shear matrix, and a pure shear is half rotation and half symmetric stretch.
BICEP's own Jacobian is $\mathrm{diag}(\cos\delta_0/\cos\delta,\,1)$, symmetric, so its intrinsic rotation vanishes; but its grid axes are the local meridian and parallel, which rotate relative to a fixed frame by the meridian convergence $\simeq (\lambda-\lambda_0)\sin\delta$.
Against the fixed-frame fiducial the two come out **[num]**
$$
\omega_{\rm SFL} \simeq \tfrac12(\lambda-\lambda_0)\abs{\sin\delta}, \qquad
\omega_{\rm BICEP} \simeq (\lambda-\lambda_0)\abs{\sin\delta} ,
$$
verified to better than 1% at $10^\circ$ east of the SPT field centre ($\delta_0=-56^\circ$): $t/2 = 0.0724$ against a measured $\omega_C = 0.0731$, and $0.145$ against $0.145$.
BICEP's curl is twice SFL's, and it is essentially *all* of BICEP's east–west distortion — BICEP's shear in the pure-east direction is only $2.4\times10^{-3}$ there.

Over the SPT 1500d footprint ($\pm50^\circ$ RA, $-70^\circ$ to $-42^\circ$, flat weighting) **[num]**:

| projection | $\kappa_{\rm rms}$ | $\abs{\gamma}_{\rm rms}$ | $\omega_{\rm rms}$ | $\abs{\vb*{d}}_{\rm rms}$ | $\abs{\vb*{d}}_{\max}$ |
|---|---|---|---|---|---|
| SFL   | 0.097 | 0.221 | **0.252** | $4.9^\circ$ | $12.7^\circ$ |
| BICEP | 0.132 | 0.120 | **0.461** | $6.1^\circ$ | $17.1^\circ$ |
| SIN   | 0.153 | 0.083 | $0$ | $1.8^\circ$ | $7.1^\circ$ |
| ZEA   | 0.110 | 0.058 | $0$ | $1.3^\circ$ | $5.5^\circ$ |
| STG   | 0.072 | 0.037 | $0$ | $0.9^\circ$ | $3.9^\circ$ |

(All referred to TAN, so $\kappa$ and $\abs{\gamma}$ include TAN's own distortion — that is why STG, whose intrinsic shear is zero, shows $\abs{\gamma}_{\rm rms}=0.037$. Independently, each projection's *own* shear over the same footprint gives $2\langle g^2\rangle = 7.8\times10^{-2}$ (SFL), $2.2\times10^{-2}$ (BICEP), $2.2\times10^{-3}$ (SIN), $5.2\times10^{-4}$ (ZEA), $10^{-13}$ (STG) — reproducing the paper's $7.0\times10^{-2}$ and $1.7\times10^{-2}$ up to the flat-versus-$W^2$ weighting **[num]**.)

**Why this is interesting.** Curl-mode lensing reconstruction is a standard systematics null: gravitational lensing produces curl only at second order in the deflection (post-Born, field rotation), at $\Omega \sim 10^{-5}$ — four to five orders below what SFL and BICEP write into the map. The rule is simple and actionable: *a projection is curl-free iff its grid axes are anchored to a single point rather than to the local meridian.* Zenithal projections pass; cylindrical and pseudocylindrical ones do not.

The reprieve is the $L$-space localization of Section 3: the curl lives at $L\lesssim10$, so with a computed mean field subtracted, the curl null test at $L\gtrsim30$ survives. The low-$L$ curl bandpowers do not.

---

## 2. Quadratic estimator response and the cartographic mean field

### The mean field is the projection's own potential

In the flat sky, for a fixed deflection $\vb*{d} = \grad\phi + \curl\Omega$ and $\vb*{L}=\vbell_1+\vbell_2$,
$$
\expval{T(\vbell_1)T(\vbell_2)}_{\rm CMB} = f^{\phi}_{\vbell_1\vbell_2}\,\phi(\vb*{L}) + f^{\Omega}_{\vbell_1\vbell_2}\,\Omega(\vb*{L}),
$$
$$
f^{\phi}_{\vbell_1\vbell_2} = C^{TT}_{\ell_1}(\vb*{L}\cdot\vbell_1) + C^{TT}_{\ell_2}(\vb*{L}\cdot\vbell_2), \qquad
f^{\Omega}_{\vbell_1\vbell_2} = C^{TT}_{\ell_1}(\vb*{L}\times\vbell_1)_z + C^{TT}_{\ell_2}(\vb*{L}\times\vbell_2)_z ,
$$
and the two kernels are orthogonal under the QE weights, so the gradient estimator is blind to $\Omega$ at linear order and vice versa.

The cartographic distortion is realization-independent.
So averaging the QE over CMB realizations at fixed projection gives, with the standard normalization,
$$
\boxed{\;\expval{\hat\phi(\vb*{L})}_{\rm CMB} = \phi_{\rm cart}(\vb*{L}), \qquad \expval{\hat\Omega(\vb*{L})}_{\rm CMB} = \Omega_{\rm cart}(\vb*{L})\;}
$$
to linear order in the distortion.
This is a *mean field* in exactly the technical sense of the mask and anisotropic-noise mean fields — with the crucial difference that it needs no simulations. The projection definition gives $J$ analytically; $J$ gives $\vb*{d}$; a Helmholtz decomposition on the grid gives $\phi_{\rm cart}$ and $\Omega_{\rm cart}$ to machine precision.

The $\ell\gg L$ limit makes the content transparent. Locally the QE measures the anisotropic modulation of the small-scale power, i.e. it responds to $\kappa_{\rm eff}$ and $\gamma_{\rm eff}$ with the standard lensing response; $\phi_{\rm cart}$ is just the potential whose Hessian those are. The paper already establishes the locality condition and its failure order: the correction is $\order{(L/\ell)^2}$, and with $L\lesssim10$ and $\ell\gtrsim10^3$ that is $\lesssim10^{-4}$ in amplitude.

### Size, and the right comparison

The naive comparison — deflection against deflection — is dramatic and slightly misleading.

Deflections of the zenithal projections, radial, in arcmin **[num]**:

| $r$ | SIN vs ARC | ZEA vs ARC | STG vs ARC | SIN vs TAN | ZEA vs TAN | STG vs TAN |
|---|---|---|---|---|---|---|
| $10^\circ$ | 3.0 | 0.8 | 1.5 | 9.2 | 6.9 | 4.6 |
| $15^\circ$ | 10.2 | 2.6 | 5.2 | 31.4 | 23.7 | 16.0 |
| $20^\circ$ | 24.2 | 6.1 | 12.3 | 75.5 | 57.3 | 38.9 |
| $25^\circ$ | 47.1 | 11.9 | 24.3 | 150.2 | 114.9 | 78.8 |

Small-$r$ forms: $r-\sin r \simeq r^3/6$ (SIN), $r-2\sin\tfrac r2 \simeq r^3/24$ (ZEA), $2\tan\tfrac r2 - r \simeq r^3/12$ (STG), $\tan r - r \simeq r^3/3$ (TAN).

Chart choice moves these by a factor of a few, so the honest, chart-free statement is the *difference between two projections*: reprojecting a $25^\circ$-radius field between SIN and ZEA and treating both grids as flat displaces the sky by **35 arcmin** at the edge **[num]**, fourteen times the gravitational deflection rms of $\approx 2.5'$. On the 1500d field the full cartographic deflection is $1$–$6^\circ$ rms depending on the projection: 25–150× gravitational.

That comparison overstates the danger, and the reason is worth stating plainly. A deflection field constant across the map is a translation and is unobservable in any $|\vbell|$ statistic; the QE's response to $\vb*{d}$ at $L\to0$ vanishes. What couples is the *gradient* of the deflection — $\kappa$ and $\gamma$. There the two are comparable:
$$
\kappa_{\rm cart}, \abs{\gamma_{\rm cart}} \sim 0.03\text{–}0.22 \quad\text{versus}\quad \kappa^{\rm grav}_{\rm rms}\approx 0.03 .
$$
Cartographic lensing is a lens with a hundred times the deflection and roughly one to seven times the shear, all of it packed into a handful of modes at $L\lesssim10$.
In potential terms, $\phi_{\rm cart}\sim \abs{\vb*{d}}/L \sim 0.05/5 \sim 10^{-2}$ against $\phi_{\rm grav}\sim 10^{-5}$ — a mean field three orders of magnitude above the signal, at the $L$ where it lives.

---

## 3. Residual biases on $C_L^{\phi\phi}$ after mean-field subtraction

The mean field is exactly computable, so subtract it. What survives is the interesting question. Four terms, in descending size.

### (a) Response and normalization error — the largest, and not a lensing bias

The QE normalization $A_L$ and the inverse-variance filters both assume a single isotropic $C_\ell$.
The projection violates that twice over.

*The isotropic part is enormous.* The paper measures the projected-to-input $EE$ ratio running from $0.665$ at $\ell=1000$ to $1.31$ at $\ell=3000$ for SFL.
Feeding unprojected theory spectra into $A_L$ therefore misnormalizes $\hat\phi$ at the tens-of-percent level and $C_L^{\phi\phi}$ at twice that.
This is trivially fixed by using the measured spectra — which everyone does — so it is a trap rather than a bias.

*The anisotropic part is not fixable that way, and it is the real term.*
The local spectrum distortion is set by the local magnification and shear, both of which vary across the footprint.
For a spectrum of local slope $n_{\rm eff}=\dd\ln C_\ell/\dd\ln\ell$, a local convergence $\kappa$ shifts the observed power by
$$
\frac{\delta C_\ell}{C_\ell} \simeq -n_{\rm eff}\,\kappa_{\rm eff} \qquad\text{(monopole)},
$$
and a local shear $\gamma$ adds a quadrupole in $\hat\vbell$ of comparable amplitude, $\sim n_{\rm eff}\abs{\gamma}$.
On the damping tail $n_{\rm eff}\approx -4$ to $-6$.
With $\kappa_{\rm eff}$ and $\abs{\gamma_{\rm eff}}$ of $0.1$–$0.22$ over the 1500d field, the local response error is
$$
\rho \equiv \frac{\delta R}{R} \sim n_{\rm eff}\,\kappa \sim 0.4\text{–}1.0 \quad\text{(SFL, BICEP on 1500d)},
$$
$$
\rho \sim 0.05\text{–}0.15 \quad\text{(ZEA, and small footprints generally)}.
$$
A multiplicative field $\rho(\vb*{x})$ on $\hat\phi$ biases the power spectrum by $\expval{\rho^2}$ *at all $L$* — it is not confined to low $L$, because the multiplication convolves.
Order of magnitude: **tens of percent for SFL/BICEP on a $1500\,\mathrm{deg}^2$ field, of order a percent for ZEA.** **[unverified: the $\rho\to\delta C_L^{\phi\phi}$ transfer needs an actual normalization calculation; I am quoting $\expval{\rho^2}$ as the scale, which is right to a factor of a few at most.]**

This is the term I would flag hardest. It is bigger than every genuinely lensing-flavoured bias below, it scales as the *first* power of the distortion, and its cure is not mean-field subtraction but simulations that include the projection.

### (b) $N^{(1)}$-like coupling between cartographic and gravitational lensing

$N^{(1)}$ is the $\order{C^{\phi\phi}}$ bias from CMB four-point contractions in which the two lens legs carry different momenta.
Replacing the lens power by the total, $C^{\phi\phi,\rm tot}_{L'} = C^{\phi\phi,\rm grav}_{L'} + \abs{\phi_{\rm cart}(\vb*{L}')}^2$, gives the cartographic contribution
$$
\delta N^{(1)}(L) = \int\!\frac{\dd[2]{\vb*{L}'}}{(2\pi)^2}\; \abs{\phi_{\rm cart}(\vb*{L}')}^2\, K(\vb*{L},\vb*{L}') .
$$
The two structural facts pull in opposite directions.
Against it: $\abs{\phi_{\rm cart}}^2$ is confined to $L'\lesssim10$, where $K$ is suppressed — $K$ must vanish as $L'\to0$, since a uniform deflection is a translation.
For it: the correct low-$L'$ measure of $K$ is not $\phi$ but $\kappa,\gamma$, and by that measure the cartographic lens is comparable to or several times the gravitational one, so no large suppression is guaranteed.

The naive ratio, distortion variance against distortion variance, is
$$
\frac{\expval{\kappa^2+\abs{\gamma}^2}_{\rm cart}}{\expval{\kappa^2+\abs{\gamma}^2}_{\rm grav}} \sim \frac{(0.03\text{–}0.22)^2}{(0.03)^2} \sim 1\text{–}50 ,
$$
and with $N^{(1)}/C_L^{\phi\phi}\approx$ a few percent at $L\sim500$ for an SPT-like configuration, that puts the cartographic term anywhere from a few percent to order unity.
**[unverified — this is the weakest number in the note. It needs $K(\vb*{L},\vb*{L}')$ evaluated at $L'\lesssim10$, which is exactly the regime where my scaling argument is least trustworthy. Do not put a number in the paper without computing it.]**

The mitigation is the same as (a): the cartographic lens is known, so include it in the $N^{(1)}$ simulations and the term is subtracted along with the gravitational one.

### (c) $N^{(0)}$

$N^{(0)}$ is quadratic in the observed total spectra.
A tens-of-percent error in $C_\ell^{\rm obs}$ propagates to a tens-of-percent error in $N^{(0)}$, which is fatal — $N^{(0)}$ dominates the raw estimator variance at all but the lowest $L$.
Using measured spectra removes the mean shift, leaving a residual of order $\expval{(\delta C_\ell/C_\ell)^2}$ across the footprint, i.e. $\expval{\rho^2}$ again, so this collapses into term (a) numerically.
Realization-dependent $N^{(0)}$ absorbs most of what remains, because it reads the spectra off the data itself.
I would not treat (c) as an independent concern in a pipeline that already uses RDN0.

### (d) The structural point about $L$ separation, and its limit

Cartographic power lives at $L\lesssim L_{\max}\approx\pi/\theta_{\rm field}$: about $7$ for a $25^\circ$-radius disk, $4$–$13$ for the 1500d field depending on direction.
Science lensing lives at $L\gtrsim30$.
So the *direct* cartographic contribution to the reconstructed spectrum — the mean-field power $\abs{\phi_{\rm cart}(L)}^2$, and the second-order term built from two cartographic modes — is confined to $L\lesssim 2L_{\max}\approx20$.
That is a genuine protection, and it is why nobody has been bitten by this yet.

It does not extend to (a) or (b).
Both are couplings of the cartographic field to *CMB* modes at $\ell\sim10^3$, not to other cartographic modes, so the sum $\vb*{L} = \vb*{L}_{\rm cart} + \vbell$ reaches anywhere.
The suppression of those terms is $\order{g^2}\sim 10^{-3}$–$5\times10^{-2}$ over the range this paper studies, and nothing more.
Stated compactly: **the $L$-separation protects the reconstructed spectrum's low-$L$ band from direct contamination, but the science band is protected only by $g^2$.**

---

## 4. The $EB$ loop: leaked $B$ is a template, not noise

### The $EB$ estimator reconstructs the cartographic shear

The lensing $EB$ estimator exists precisely because a lens modulates $E\to B$ conversion:
$$
\expval{E(\vbell_1)B(\vbell_2)}_{\rm CMB} = \phi(\vb*{L})\; C^{EE}_{\ell_1}\,(\vb*{L}\cdot\vbell_1)\,\sin 2(\varphi_{\ell_1}-\varphi_{\ell_2}), \qquad \vb*{L}=\vbell_1+\vbell_2 .
$$
The paper's $BB = 2\expval{g^2}EE$ is the azimuthally averaged, constant-shear limit of the same coupling — the paper already establishes that the $E\to B$ lensing kernel reduces to the leakage relation.
Run the $EB$ estimator on a projected, unpurified map and it returns the cartographic shear field:
$$
\expval{\hat\gamma^{EB}(\vb*{L})}_{\rm CMB} = \gamma_{\rm cart}(\vb*{L}) .
$$
For SFL on the 1500d field, $\abs{\gamma_{\rm cart}}_{\rm rms}\approx 0.2$ against a gravitational $0.03$: the $EB$ reconstruction at $L\lesssim10$ is dominated by the projection by a factor $\sim50$ in power.

So the leaked $B$ modes are not a noise floor. They are a deterministic image of the observed $E$ modes under a known linear operator, and that changes what can be done about them.

### Template subtraction, and why it works better than delensing

Write $\mathcal{S}_J$ for the shear-leakage operator: the linear map that takes $E$ on the grid to the $B$ the projection produces, fixed entirely by $J$.
In the principal frame of $\Sigma$ it is multiplication by $\sin2\delta\psi(\chi)$ with $\delta\psi = \arctan[(s_2/s_1)\tan\chi]-\chi$; in the map domain it is a position-dependent spin-2 mixing with known coefficients.
Given the measured $\hat E = E + N_E$, build $B_{\rm temp} = \mathcal{S}_J[\hat E]$ and subtract.
The residual is $\mathcal{S}_J[N_E]$, with power
$$
\boxed{\;C_\ell^{BB,\rm resid} = 2\expval{g^2}\, N_\ell^{EE}\;}
$$
so the fractional leakage remaining is $N_\ell^{EE}/C_\ell^{EE}$.
For an SPT-3G-like $5\,\mu\mathrm{K}\text{-arcmin}$ polarization map at $\ell=1000$ ($\mathcal{D}_\ell^{EE}\approx 40\,\mu\mathrm{K}^2$), that ratio is $\approx 0.8\%$.
SFL's $BB/EE = 7\times10^{-2}$ on the 1500d field — thirty times the lensed-$B$ floor of $2.3\times10^{-3}$ — becomes $\approx 6\times10^{-4}$, a quarter of the floor.
**Two orders of magnitude, without changing the projection.** **[the $N^{EE}/C^{EE}$ number is an order-of-magnitude plug from a nominal noise level, not an SPT-3G survey specification]**

This is the delensing logic with the difficulty removed.
Delensing needs $\phi$ reconstructed from noisy data, so the template is limited by reconstruction noise and the residual is set by how much of $\phi$ you know.
Here the lens is known exactly, so the template is limited only by the noise on $E$ — and $E$ is the highest signal-to-noise field a CMB polarization experiment has.
It is also the FRANK construction (project an $E$-only sky, read off the induced $B$) promoted from a diagnostic to a correction: FRANK measures $\mathcal{S}_J[E_{\rm sim}]$ on simulations, and the same operator applied to the *data's* $E$ is the template.

Two honest limits.
The $E$ used in the template must itself be free of $E\to B$ ambiguity from the mask, so the operation belongs after purification, or the mask leakage and the shear leakage have to be handled together.
And $\mathcal{S}_J$ as written is the $\order{g^2}$ operator; at $g=0.2$ the exact kernel's $-\tfrac34 g^2$ correction is a $3\%$ effect on the template, well above the $0.8\%$ residual floor, so the exact $\sin2\delta\psi$ kernel is needed, not the leading term.

---

## 5. Beyond two-point, in general

Get the deterministic-versus-stochastic distinction exactly right, because it decides whether this is a non-Gaussianity problem or an isotropy problem.

Over the CMB ensemble at fixed projection, the projected map is a *linear* transform of a Gaussian field, so it is exactly Gaussian.
Every connected $N$-point function vanishes, for all $N$.
There is no cartographic bispectrum and no cartographic trispectrum.
What the projection breaks is statistical isotropy: $\expval{\tilde T(\vbell_1)\tilde T(\vbell_2)}$ is nonzero for $\vbell_1+\vbell_2 = \vb*{L}\neq0$, with $\vb*{L}$ restricted to $L\lesssim10$.
The whole signature lives in the two-point function, off the diagonal.

An analysis that assumes isotropy converts that into an apparent four-point signal.
Estimating the connected trispectrum by subtracting the isotropic-Gaussian disconnected piece leaves exactly $\abs{\expval{\tilde T\tilde T}_{\rm off-diag}}^2$, which has the lensing trispectrum's form with $C_L^{\phi\phi}\to\abs{\phi_{\rm cart}(\vb*{L})}^2$.
That is the mean-field power of Section 2, seen from the four-point side, and it is the same object — the two descriptions are not independent effects.

One property distinguishes it from a real lensing signal in the data itself, without any external knowledge.
A stochastic lensing field averages down over realizations; a deterministic one does not.
The apparent trispectrum is a single realization, so mode by mode it fluctuates as a $\chi^2$ with one degree of freedom rather than shrinking as $1/\sqrt{N_{\rm modes}}$ within a bandpower.
Seed-to-seed, it is perfectly correlated — which is the same diagnostic the paper already uses at the spectrum level, where STG's floor shows off-diagonal $BB$ correlations approaching unity and thereby reveals itself as a seed-independent template rather than a sky.

---

## Candidates for the paper

Ordered by how confident I am and how well they fit the existing manuscript.

**Strong — I would put these in.**

1. *Zenithal projections are exactly curl-free; SFL and BICEP are not.* One short subsection or a long paragraph in the Discussion. The closed-form potentials $\phi_{\rm SIN},\phi_{\rm ZEA},\phi_{\rm STG}$ fit in two lines, the criterion ("grid axes anchored to a point, not to the meridian") is a one-sentence rule, and the payoff is concrete: the curl estimator is a standard systematics null, gravitational curl is $\sim10^{-5}$, and SFL/BICEP write $\omega_{\rm rms}\sim0.25$–$0.46$ into the map. This is a genuinely new statement and it is fully derived. It also strengthens the paper's existing recommendation of ZEA with an argument that has nothing to do with shear.

2. *The leaked $B$ is a template, and template subtraction buys two orders of magnitude.* The $2\expval{g^2}N_\ell^{EE}$ residual formula is a two-line derivation and it converts the paper's headline law from a diagnostic into a correction. It also connects the paper to FRANK and to delensing, which the Discussion currently does not do. Worth a subsection.

3. *One paragraph in the Discussion on lensing reconstruction*, making the structural point: the projection is a mean field at $L\lesssim10$, computable in advance without simulations, and the science band at $L\gtrsim30$ is protected not by the $L$ separation but by $g^2$. This is the natural extension of the paper's "deterministic and computable a priori" theme into the four-point domain.

4. *The deterministic-versus-stochastic paragraph* (Section 5). Short, clean, and it forestalls the obvious referee question ("does this make the map non-Gaussian?"). The answer — Gaussian but anisotropic, masquerading as a trispectrum only under an isotropy assumption — is exact and costs three sentences.

**Worth a sentence, with the number left out.**

5. *The response-error term (3a).* The qualitative statement is safe and important: the QE normalization assumes one isotropic spectrum, and the projection makes the local spectrum vary by tens of percent across a large footprint, so lensing pipelines must build the projection into their simulations rather than treat it as a spectrum-level correction. I would say that much and not quote $\expval{\rho^2}$.

**Do not put in the paper without a calculation.**

6. *The $N^{(1)}$ cross-coupling (3b).* The mechanism is right and the framing (distortion variance, not deflection variance) is right, but my number spans "few percent" to "order unity" and the kernel behaviour at $L'\lesssim10$ is exactly what I have not computed. Either compute it or leave it as an explicitly open item.

**Things I checked that turned out not to be true, worth not re-deriving.**

- BICEP's own Jacobian is symmetric, so its *intrinsic* rotation is zero. Its large curl comes entirely from the meridian convergence of its grid axes against a fixed frame. Both statements are true and they are easy to confuse.
- $\kappa_{\rm eff}$ and the deflection magnitude are fiducial-chart-dependent at exactly the order of the effect itself. Any quoted deflection needs its reference chart named. The chart-free quantities are $g$ (from $\Sigma$), $\omega$ (against any point-anchored chart), and differences between projections.

---

## Appendix: numerical checks

`notes/curl.py` and `notes/own.py` (small, self-contained, `uv run`-able with numpy only).
They build each projection from its Calabretta & Greisen (2002) definition, take finite-difference Jacobians between the local orthonormal sky frame and the grid, and evaluate $\kappa,\gamma_1,\gamma_2,\omega$ for the projection alone and for the composite against TAN.
Checks that passed:

- $\omega < 10^{-10}$ for SIN, ZEA, STG at every offset out to $25^\circ$ — the finite-difference floor, confirming exact curl-freedom.
- SFL: $\omega_C = 0.0731$ against the analytic $t/2 = 0.0724$ at $10^\circ$ east; BICEP: $0.145$ against $(\lambda-\lambda_0)\sin\delta = 0.1447$.
- STG's own shear $\langle g^2\rangle = 6\times10^{-14}$, the derivative floor, consistent with the paper.
- Own-shear $2\langle g^2\rangle$ over the 1500d field reproduces the paper's SFL and BICEP values to within the flat-weight versus $W^2$-weight difference.
