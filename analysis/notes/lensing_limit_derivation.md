# The shear law as the constant-deflection-gradient limit of CMB lensing

**Question.** The paper states $C_\ell^{BB}/C_\ell^{EE} = 2\langle g^2\rangle$, with the exact
constant-shear form $2g^2(1-\tfrac34 g^2)$, $g=(s_1-s_2)/(s_1+s_2)$. A referee who
knows lensing will immediately try to recover this from the standard flat-sky
$E\to B$ lensing kernel. Does that route give the coefficient **2** exactly?

**Answer: yes, exactly.** The first-order lensing kernel, evaluated in the limit of a
deflection field whose gradient is constant across a mode ($L \ll \ell$), returns
$C_\ell^{BB} = 2\langle|\gamma|^2\rangle\, C_\ell^{EE}$, and the lensing shear modulus
$|\gamma|$ maps onto our $g$ as $g = |\gamma|/(1-\kappa)$, i.e. $g = |\gamma|$ at first
order. The two derivations agree at $O(g^2)$ with no residual factor. They must
differ at $O(g^4)$, because the lensing kernel is first order in $\phi$ while our
result is non-perturbative — and indeed the $-\tfrac34 g^4$ term is invisible to the
first-order kernel. This is a genuinely independent validation: different machinery
(perturbative convolution in deflection power) and different averaging (over
deflection-mode direction rather than over mode azimuth) landing on the same 2.

---

## 1. The direct (exact, non-perturbative) route

### 1.1 Setup

Let the projection's local Jacobian be $J = RS$ (polar decomposition, $R$ a rotation,
$S$ symmetric positive definite). `FlattenPol` removes exactly $R$
(paper Eq. 8 = the polar rotation), so the effective map is the symmetric $S$, with
eigenvalues $s_1 \ge s_2 > 0$ along principal axes. Work in the principal frame and
take $S$ constant (the adiabatic limit; see §3).

Rotation-only flattening means the grid polarization is the *sky* polarization,
components untouched in the sky basis, evaluated at the remapped position:

$$ P_{\rm grid}(\mathbf{x}) \;=\; P_{\rm sky}(S\mathbf{x}), \qquad P \equiv Q + iU. $$

This is structurally identical to the CMB lensing remapping $P(\mathbf{x}) =
\tilde P(\mathbf{x}+\mathbf{d}(\mathbf{x}))$, in which the Stokes components are
likewise carried unrotated in the flat-sky basis. That structural identity is what
makes the comparison in §2 legitimate.

### 1.2 Fourier transform

With $\mathbf{X} = S\mathbf{x}$ and $S = S^{T}$,

$$ \tilde P_{\rm grid}(\boldsymbol\ell) = \frac{1}{\det S}\,\tilde P_{\rm sky}(\mathbf{q}),
\qquad \mathbf{q} = S^{-1}\boldsymbol\ell . $$

So grid mode $\boldsymbol\ell$ is fed by *one* sky mode $\mathbf{q}$ — a constant linear
remap does not mix $|\ell|$ shells with each other, it just relabels them. (With the
opposite Jacobian convention, $\mathbf{q}=J^{T}\boldsymbol\ell$ as written in the paper;
this only flips the sign of $g$, and everything below depends on $g^2$.)

### 1.3 E/B

With the flat-sky convention $E(\boldsymbol\ell)+iB(\boldsymbol\ell) =
-\tilde P(\boldsymbol\ell)\,e^{-2i\varphi_{\boldsymbol\ell}}$, and $B_{\rm sky}=0$:

$$ E_{\rm grid}(\boldsymbol\ell) + i B_{\rm grid}(\boldsymbol\ell)
= \frac{1}{\det S}\,E_{\rm sky}(\mathbf{q})\, e^{2i\,\delta\psi},
\qquad \delta\psi \equiv \varphi_{\mathbf q} - \varphi_{\boldsymbol\ell}. $$

Hence, mode by mode,

$$ E_{\rm grid} = \tfrac{1}{\det S} E_{\rm sky}(q)\cos 2\delta\psi, \qquad
B_{\rm grid} = \tfrac{1}{\det S} E_{\rm sky}(q)\sin 2\delta\psi, $$

and the ratio is **exactly** $\tan^2 2\delta\psi$ per mode; after annulus binning
(both numerator and denominator read $E_{\rm sky}$ at the same $q$, and $\det S$
cancels),

$$ \frac{C_\ell^{BB}}{C_\ell^{EE}} = \frac{\langle \sin^2 2\delta\psi\rangle_\chi}
{\langle \cos^2 2\delta\psi\rangle_\chi} \;\simeq\; \langle\sin^2 2\delta\psi\rangle_\chi . $$

(The denominator is $1-O(g^2)$; keeping it exactly would give
$\langle\sin^2\rangle/(1-\langle\sin^2\rangle)$, a further $O(g^4)$ refinement that the
paper's quoted form does not include. Both are $O(g^4)$, i.e. below the 1.7%
correction level already flagged.)

### 1.4 The azimuthal average in closed form

Write $\boldsymbol\ell = \ell(\cos\chi,\sin\chi)$ in the principal frame. Then
$\mathbf{q}\propto(s_2\cos\chi,\ s_1\sin\chi)$ (after multiplying by $s_1s_2$), so with
$\Sigma=(s_1+s_2)/2$, $\Delta=(s_1-s_2)/2$, $g=\Delta/\Sigma$:

$$ \sin 2\delta\psi = \frac{2g\sin 2\chi\,(1-g\cos2\chi)}{1+g^2-2g\cos2\chi}. $$

Let $\theta = 2\chi$, $w=e^{i\theta}$, $z = 1-gw$. Then $g\sin\theta = -\operatorname{Im}z$
and $1-g\cos\theta = \operatorname{Re}z$, so the numerator is
$\big(\operatorname{Im}z^2\big)^2$ and the denominator $|z^2|^2$:

$$ \sin^2 2\delta\psi = \sin^2\!\big(2\arg z\big), \qquad z = 1 - g e^{i\theta}. $$

With $e^{4i\alpha} = (z/\bar z)^2$ for $\alpha=\arg z$, the average is a contour integral
with a **double pole at $w=g$** inside the unit circle:

$$ \langle\cos 4\alpha\rangle = \frac{1}{2\pi i}\oint \frac{w(1-gw)^2}{(w-g)^2}\,dw
= \left.\frac{d}{dw}\big[w(1-gw)^2\big]\right|_{w=g} = (1-g^2)(1-3g^2). $$

Therefore

$$ \boxed{\ \langle\sin^2 2\delta\psi\rangle = \tfrac12\big[1-(1-g^2)(1-3g^2)\big]
= 2g^2 - \tfrac32 g^4 = 2g^2\Big(1-\tfrac34 g^2\Big).\ } $$

Verified numerically to machine precision for $g \in \{0.01,\dots,0.7\}$
(`analysis/notes/` check; e.g. $g=0.4$: quadrature $0.2816000$, formula $0.2816000$).

The coefficient 2 is $4\times\langle\sin^2 2\chi\rangle = 4\times\tfrac12$: the leading
misalignment is $\delta\psi \simeq -g\sin2\chi$, and the azimuthal average of
$\sin^2 2\delta\psi \simeq 4g^2\sin^2 2\chi$ supplies the $\tfrac12$.

---

## 2. The lensing-kernel route

### 2.1 The standard kernel

Flat-sky, first order in the lensing potential $\phi$ (Hu 2000; Lewis & Challinor 2006,
Phys. Rep. 429, §4.2), for an unlensed $B$-free sky:

$$ C_\ell^{BB} = \int\!\frac{d^2\boldsymbol\ell'}{(2\pi)^2}
\big[\boldsymbol\ell'\!\cdot(\boldsymbol\ell-\boldsymbol\ell')\big]^2\,
C^{\phi\phi}_{|\boldsymbol\ell-\boldsymbol\ell'|}\, C^{EE}_{\ell'}\,
\sin^2 2(\varphi_{\boldsymbol\ell'}-\varphi_{\boldsymbol\ell}). $$

### 2.2 The constant-gradient limit

A strictly constant deflection gradient, $\mathbf d(\mathbf x) = (S-I)\mathbf x$, i.e.
$\phi = \tfrac12 \mathbf x^{T}(S-I)\mathbf x$, is not a stationary random field: in
Fourier it is a $\delta''$ at $L=0$. The clean regularization — and the one that matches
the physical situation, a shear that varies slowly across the footprint — is to keep
$C^{\phi\phi}_L$ supported at $L \ll \ell$ and expand. Substituting
$\boldsymbol\ell' = \boldsymbol\ell - \mathbf L$:

- $\boldsymbol\ell'\!\cdot\mathbf L \to \boldsymbol\ell\cdot\mathbf L = \ell L_\parallel$,
- $C^{EE}_{|\boldsymbol\ell-\mathbf L|}\to C^{EE}_\ell$,
- the angle between $\boldsymbol\ell-\mathbf L$ and $\boldsymbol\ell$ is
  $-L_\perp/\ell$, so $\sin^2 2(\varphi_{\ell'}-\varphi_\ell) \to 4L_\perp^2/\ell^2$,

where $L_\parallel, L_\perp$ are the components of $\mathbf L$ along and across
$\boldsymbol\ell$. The two $\ell^2$'s cancel and

$$ C_\ell^{BB} = 4\,C_\ell^{EE}\int\!\frac{d^2\mathbf L}{(2\pi)^2}\,
L_\parallel^2 L_\perp^2\, C^{\phi\phi}_L . $$

Numerical check of the expansion (exact angular integral vs. its $L\ll\ell$ limit):
the ratio is $1 + 0.50\,(L/\ell)^2 + \ldots$ — $1.0005$ at $L/\ell=0.01$, $1.0050$ at
$0.1$, $1.0196$ at $0.2$. **The adiabatic correction is $O((L/\ell)^2)$, not
$O((L/\ell))$** — see §3.

### 2.3 Recognizing the shear

With $\gamma_1 = \tfrac12(\partial_1^2-\partial_2^2)\phi$, $\gamma_2=\partial_1\partial_2\phi$
in a frame whose 1-axis is $\hat{\boldsymbol\ell}$, the cross component of the shear
*relative to the mode direction* is $\gamma_\times = \gamma_2$, and
$\tilde\gamma_2(\mathbf L) = -L_\parallel L_\perp\,\tilde\phi(\mathbf L)$. So the integral
above is precisely $\langle\gamma_\times^2\rangle$:

$$ C_\ell^{BB} = 4\,\langle\gamma_\times^2\rangle\,C_\ell^{EE}. $$

For a statistically isotropic shear field (or, equivalently here, for a fixed shear with
the mode azimuth averaged over the annulus), $\gamma_\times = -|\gamma|\sin 2\chi$ and
$\langle\gamma_\times^2\rangle = \tfrac12\langle|\gamma|^2\rangle$. Hence

$$ \boxed{\ \frac{C_\ell^{BB}}{C_\ell^{EE}} = 2\,\langle|\gamma|^2\rangle\ } $$

— coefficient **2**, exactly, from the standard lensing kernel.

Note the structural echo: the kernel's $\sin^2 2\varphi_{\ell\ell'}$ *is* our
$\sin^2 2\delta\psi$. In the lensing calculation the misalignment is between
$\boldsymbol\ell$ and the source mode $\boldsymbol\ell'=\boldsymbol\ell-\mathbf L$; in ours
it is between $\boldsymbol\ell$ and $\mathbf q = S^{-1}\boldsymbol\ell$. They are the same
angle, computed to first and to all orders respectively. The factor 2 is $4\times\tfrac12$
in both derivations, with the $\tfrac12$ coming from $\langle\sin^2 2\chi\rangle$ either
way.

---

## 3. Convention mapping

| quantity | lensing convention | this paper | relation |
|---|---|---|---|
| remap | $\mathbf x \mapsto \mathbf x + \nabla\phi$, Jacobian $A = I + \nabla\nabla\phi$ | $\mathbf x \mapsto S\mathbf x$, $S$ symmetric | $S \leftrightarrow A$ |
| convergence | $\kappa = \tfrac12\nabla^2\phi$ | $(s_1+s_2)/2 = 1+\kappa$ | magnification only; cancels in $BB/EE$ |
| shear | $\gamma = \gamma_1+i\gamma_2$, $|\gamma| = \tfrac12(s_1-s_2)$ | $g = \dfrac{s_1-s_2}{s_1+s_2}$ | $g = \dfrac{|\gamma|}{1+\kappa}$; $g=|\gamma|$ at first order |
| singular values | $s_{1,2} = 1+\kappa\pm|\gamma|$ | $s_1 \ge s_2$ | same |
| spin | $\gamma$ is spin-2, rotates as $e^{2i\varphi}$ | $\delta\psi$ enters as $\sin 2\delta\psi$ | consistent |
| symmetry | $A$ symmetric by construction (gradient deflection) | $J$ general; `FlattenPol` removes the polar rotation $R$, leaving symmetric $S$ | **equivalent only because $R$ is removed** |
| transport | Stokes components carried unrotated in the flat basis | rotation-only flattening | same structure |

Two convention traps worth naming explicitly, since they are where a referee's
thirty-second check can go wrong:

1. **$g$ is $|\gamma|$ normalized by the mean magnification, not by unity.** The
   difference is $O(\kappa g)$ and is invisible at first order in lensing but real for
   BICEP's projection, whose magnification reaches $3.9\times$. Our $g$ is the correct
   variable — it is scale-free, as it must be, since an overall dilation cannot leak.
2. **Factors of 2 in $\gamma$.** Some references define reduced shear or use
   $\gamma = \tfrac12(s_1/s_2 - 1)$-type variables. With the standard
   $|\gamma| = \tfrac12(s_1-s_2)$ used above, the coefficient is 2 and nothing else.
   The paper's table of alternative reductions (fitted $k=0.4978$ for
   $\langle\ln^2(s_1/s_2)\rangle$, i.e. $\approx\tfrac12$) is the same statement seen
   from the other side: $\ln(s_1/s_2)\simeq 2g$, and $\tfrac12(2g)^2 = 2g^2$.

## 4. Where the two derivations part company, and what that buys us

- **$O(g^2)$: identical.** Free independent validation.
- **$O(g^4)$:** the lensing kernel is first order in $\phi$ and cannot produce
  $-\tfrac32 g^4$. Our exact result can, because a constant linear remap is solved
  in closed form rather than expanded. A referee who computes only the first-order
  lensing result and finds $2g^2$ has *confirmed* the paper, not contradicted the
  $(1-\tfrac34 g^2)$ refinement.
- **Adiabaticity.** The paper currently states corrections at
  $O((\ell L)^{-1})$. The kernel expansion says they are $O((\ell L)^{-2})$ — the
  $O(L/\ell)$ term vanishes by parity in the angular integral. This is a
  strengthening, and the paper should either quote $(\ell L)^{-2}$ or keep
  $(\ell L)^{-1}$ as a conservative bound and say so. (Real footprints also carry
  mask-boundary corrections that are not this expansion.)
- **What lensing cannot supply.** The lensing kernel assumes a *gradient* deflection,
  hence symmetric $A$, hence no rotation. A projection Jacobian is generally not
  symmetric. The mapping is legitimate here only because `FlattenPol` removes exactly
  the polar rotation. A pipeline that flattened with any other angle would have a
  residual antisymmetric part, which is a *curl* deflection in lensing language, and
  would produce additional $B$ power that this derivation does not describe.

---

## 5. One-paragraph version for the paper

> The coefficient is not new physics. Equation (11) is the constant-deflection-gradient
> limit of the standard CMB lensing $E\to B$ kernel. Writing the flattened projection as
> a remapping with Jacobian $A = (1+\kappa)I + \gamma$-matrix and inserting a lensing
> potential whose power is confined to $L \ll \ell$, the first-order kernel
> $C_\ell^{BB} = \int \frac{d^2\boldsymbol\ell'}{(2\pi)^2}[\boldsymbol\ell'\!\cdot(\boldsymbol\ell-\boldsymbol\ell')]^2
> C^{\phi\phi}_{|\boldsymbol\ell-\boldsymbol\ell'|}C^{EE}_{\ell'}\sin^2 2\varphi_{\boldsymbol\ell\boldsymbol\ell'}$
> collapses to $C_\ell^{BB} = 4\langle\gamma_\times^2\rangle C_\ell^{EE}
> = 2\langle|\gamma|^2\rangle C_\ell^{EE}$, where $\gamma_\times$ is the shear component
> cross to the mode and the factor $\tfrac12$ is the azimuthal average
> $\langle\sin^2 2\chi\rangle$. Since $|\gamma| = (s_1-s_2)/2$ and our
> $g = (s_1-s_2)/(s_1+s_2)$ differ only by the mean magnification, this reproduces
> $2\langle g^2\rangle$ exactly at leading order, by entirely different machinery. The
> lensing route is perturbative in the deflection and therefore stops there; the closed
> form of Equation (11) supplies the $-\tfrac34 g^2$ correction, which is $O(\gamma^4)$
> and invisible to the first-order kernel. The same expansion also sharpens the
> adiabaticity condition: the $O(1/\ell L)$ term vanishes by parity, and the leading
> correction to locality is $O((\ell L)^{-2})$.
