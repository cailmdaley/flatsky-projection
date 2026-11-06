"""Build the thread's self-contained report from generated paper summaries."""

import base64
from datetime import datetime,timezone
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/home/cailmdaley/loom/.felt/science/spt/pol-project-paper/beams-through-reprojection/tapestry.html')


def figure(name,caption):
    p=ROOT/'figures'/name
    if not p.exists():
        return ''
    encoded=base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def run():
    data=json.loads((ROOT/'paper'/'section5_summary.json').read_text())
    beamfile=ROOT/'paper'/'beam_summary.json'
    beam=json.loads(beamfile.read_text()) if beamfile.exists() else None
    table=[]
    for key,label,note in (
        ('GattiTAN','Gatti: 14.6° gnomonic square','Uniform tangent-plane weight; idealized angular side extent.'),
        ('PiresTAN','Pires: 10° gnomonic square','Uniform tangent-plane weight; idealized angular side extent.'),
        ('HEALPixPolarFace','Jeffrey: one native polar HEALPix face','Full Nside=1 polar face; native nested square chart, rather than the actual DES mask.'),
        ('HEALPixEquatorialFace','Native equatorial HEALPix face','Shown to expose the dependence on which face is flattened.'),
        ('HEALPixPlane','Native HEALPix faces: all-sky mean','Within-face chart metric, estimated from pixel diagonals; seams excluded.')):
        g2=data['footprints'][key]
        table.append(f'<tr><th>{label}</th><td>{g2:.3g}</td><td>{2*g2:.3g}</td><td>{note}</td></tr>')
    for proj in ('ProjSFL','ProjZEA'):
        g2=data['anisotropic'][f'sky_field1500d_{proj}']['g2']
        table.append(f'<tr><th>1500d: {proj[4:]}</th><td>{g2:.3g}</td><td>{2*g2:.3g}</td><td>Production analytic C2 taper, W² sky-area weighting.</td></tr>')
    anis=[]
    for footprint,label in (('disk5','5% disk'),('field1500d','1500d'),('fieldEast','1500d eastern half')):
        for proj in ('ProjSFL','ProjZEA'):
            s=data['anisotropic'][f'sky_{footprint}_{proj}']
            g=data['anisotropic'][f'grid_{footprint}_{proj}']
            cells=[100*s['chi_zero'],100*s['chi_diagonal'],100*g['chi_zero'],100*g['chi_diagonal']]
            anis.append(f'<tr><th>{label}, {proj[4:]}</th>'+''.join(f'<td>{v:+.3f}%</td>' for v in cells)+'</tr>')
    if beam:
        rows=[]
        for name,label in (('disk5','5% disk'),('field1500d','1500d')):
            row=beam['response'][name]
            rows.append(f"<tr><th>{label}</th><td>{row['sfl_mean']:.5f}</td><td>{row['sphere_mean']:.5f}</td><td>{100*row['fractional_difference']:+.3f}%</td><td>{100*row['prediction']:+.3f}%</td><td>{100*row['sector_model_max_abs_1000_3000']:.3f} pp</td></tr>")
        beamtext=f'''<p>A one-arcminute Gaussian applied to the existing SFL Q/U map retains a different response after reprojection to ZEA than the same Gaussian applied on the sphere.
        The table compares paired filtered/unfiltered power ratios at target multipole {beam['ell_response']:.0f}; the prediction integrates the local operator over the same target Fourier bins.</p>
        <table><thead><tr><th>Footprint</th><th>SFL-filtered</th><th>Sphere-filtered</th><th>Difference</th><th>Prediction</th><th>Max sector residual, ℓ=1000–3000</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
        <p>The directional power response spans several percentage points even where its azimuthal average nearly cancels.
        This is one paired realization, not an ensemble validation: the residuals above are measured absolute differences in power response, with no fitted statistical uncertainty.
        The local prediction omits aliasing, mask convolution and Jacobian gradients; those effects have not been separated in the remaining discrepancy.
        Filtered/unfiltered ratios measure the end-to-end response, and do not in general cancel a spatially varying sampling operator exactly.</p>'''
        beamtext+=figure('beam_reprojection.png','SFL filtering retains its direction-dependent response on ZEA. The conformal STG panel compares the filter differential with the unfiltered path mismatch; these are numerical sensitivity diagnostics, not a calibrated physical B-mode upper limit.')
        brows=[]
        for name,label in (('disk5','5% disk'),('field1500d','1500d')):
            b=beam['b_diagnostic'][name+'_ProjSTG']
            brows.append(f"<tr><th>{label}</th><td>{b['sfl_base']:.3g}</td><td>{b['sphere_base']:.3g}</td><td>{b['mismatch']:.3g}</td><td>{b['differential']:.3g}</td><td>{b['cross_mismatch']:.3g}</td><td>{b['mismatch_power_change']:.3g}</td></tr>")
        beamtext+=f'''<h3>Polarization diagnostic</h3><p>A constant local Jacobian gives a common scalar Q/U multiplier and preserves sky E/B.
        The differential D=(S<sub>F</sub>−S<sub>0</sub>)−(R<sub>F</sub>−R<sub>0</sub>) also contains changed numerical transfer and reference error.
        STG removes leading target shear, but those numerical terms remain.
        The following are production-workspace, decoupled band estimates over 1000≤ℓ&lt;3000, normalized by baseline EE; they quantify the test's sensitivity, rather than isolate physical beam-induced B.</p>
        <table><thead><tr><th>STG target</th><th>SFL baseline BB/EE</th><th>Sphere baseline BB/EE</th><th>Mismatch BB/EE</th><th>Differential BB/EE</th><th>D×mismatch BB/EE</th><th>Change of mismatch BB/EE</th></tr></thead><tbody>{''.join(brows)}</tbody></table>
        <p>The mismatch-power change is C<sub>D</sub>+2C<sub>DM</sub>, where M=S<sub>0</sub>−R<sub>0</sub>.
        Its negative sign in both STG footprints means the Gaussian reduces the numerical path mismatch in this band.
        These measurements do not isolate physical E→B conversion or set a physical upper limit; the local constant-Jacobian model predicts none, while gradients and numerical errors remain unresolved.</p>
        <p>The spherical reference retains the input's numerical B residual without filtering it; an independent input check bounds that residual below 1.6×10<sup>−15</sup> of EE below ℓ=4000 and 7.0×10<sup>−14</sup> over 4000–6000.
        This E-only approximation is negligible at the measured diagnostic sensitivity.</p>'''
    else:
        beamtext='<p>The paired map experiment is running through lc. No measured beam-response or B-mode claim is available yet.</p>'
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    page=f'''<!doctype html><html><head><meta charset="utf-8"><title>Beams, anisotropic spectra, and footprint shear</title>
    <style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:3rem auto;padding:0 1.2rem;color:#222;background:#faf9f6}}h1,h2,h3{{line-height:1.2}}h2{{margin-top:2.4rem}}table{{border-collapse:collapse;width:100%;font-size:.88rem;margin:1.5rem 0}}th,td{{padding:.55rem;border-bottom:1px solid #ccc;text-align:left}}td{{font-variant-numeric:tabular-nums}}thead{{background:#eeece5}}figure{{margin:2rem 0}}img{{width:100%;height:auto}}figcaption,small{{color:#555;font-size:.9rem}}code{{font-size:.85em}}.equation{{padding:1rem;background:#eeece5;overflow:auto}}a{{color:#275d91}}</style></head><body>
    <h1>Filtering survives reprojection; anisotropy needs a declared frame</h1><small>{stamp} · Codex on behalf of Cail</small>
    <p>The anisotropic coefficient depends on where the quadrupole is defined.
    A local sky scan/TOD quadrupole gives (n+2)ga₂/2 for constant a₂, while an output-grid Fourier quadrupole gives nga₂/2.
    The footprint table measures chart shear; its spin-two benchmark does not imply leakage in scalar convergence pipelines.</p>
    <h2>Gaussian beam after reprojection</h2>{beamtext}
    <h2>Sky scan and output-grid quadrupoles</h2>
    <p>The observed power is C(ℓ,χ)=C(ℓ)[1+a₂(ℓ)cos2(χ−χ₀)], with n=d ln C/d ln ℓ and g=(s₁−s₂)/(s₁+s₂).
    For a sky-attached quadrupole, projection changes both mode length and direction.</p>
    <div class="equation">Sky scan/TOD: δC/C = (g/2)[(n+2)a₂ + da₂/d ln ℓ] cos2χ₀ + O(g²)<br>Output-grid Fourier filter: δC/C = (n/2)g a₂ cos2χ₀ + O(g²)</div>
    <p>The table uses n=−3 and constant a₂=0.3.
    The idealized scan high pass |ℓ<sub>x</sub>|&gt;300 has a₂=sin(2θ)/θ, θ=arccos(300/ℓ), giving a₂≈0.286 at ℓ=1500.
    This illustrates the scale of an SPT-like scan filter, rather than fitting an actual transfer function; <a href="https://arxiv.org/abs/2101.01684">the SPT-3G 2018 analysis</a> motivates the cutoff scale.</p>
    <table><thead><tr><th>Footprint and chart</th><th>Sky χ₀=0°</th><th>Sky χ₀=45°</th><th>Grid χ₀=0°</th><th>Grid χ₀=45°</th></tr></thead><tbody>{''.join(anis)}</tbody></table>
    {figure('anisotropic_response.png','The two anisotropy conventions differ by a factor of three at n=−3. On a symmetric footprint the 45° sky-scan response cancels; an eastern-half cut admits a first-order SFL response of about 2.35%. Solid local curves are angular quadrature and dashed curves are the leading expansion.')}
    <p>Checks include the exact sky-scan n=−2 null, the exact n=0 response ga₂, radial-a₂ and output-grid derivatives, native-grid Gaussian invariance, mask quadrature convergence at Nside 128/256, and north/south HEALPix-face symmetry.
    Independent Codex reviews rederived the coefficients and checked the table's chart definitions.</p>
    <h2>Footprint shear benchmarks</h2>
    <p>These are uniform idealizations for the literature patches, with their chart assumptions stated explicitly.
    The last column 2⟨g²⟩ is a leading spin-two shear benchmark after removing rotation; it is not a prediction of actual B power for arbitrary spectra or masks.
    <a href="https://arxiv.org/html/2310.17557v2">Gatti</a> projects scalar convergence, and <a href="https://arxiv.org/html/2403.02314v1">Jeffrey</a> reconstructs spherical E/B convergence before flattening separate scalar channels.
    Their scalar pipelines therefore do not acquire E→B leakage from these subsequent chart operations.</p>
    <table><thead><tr><th>Assumed chart/footprint</th><th>⟨g²⟩</th><th>2⟨g²⟩</th><th>Weight and scope</th></tr></thead><tbody>{''.join(table)}</tbody></table>
    <p>The gnomonic angular-side convention differs by roughly 2% from treating 14.6° as a tangent-plane length, smaller than the precision appropriate for the approximate published patch dimensions.
    The native HEALPix face estimate depends strongly on face class; it is not a reconstruction of Jeffrey's selected DES pixels.</p>
    <h2>Reproduction</h2><p><code>bash env.sh python3 analysis/paper_numbers.py --section5-only</code> writes <code>paper/section5_numbers.tex</code> and <code>paper/section5_summary.json</code>.
    The paired map recipe is <code>analysis/beams/astra.yaml</code>; run <code>lc run --executor local -u baseline beam_response</code> there.
    <code>analysis/beam_report.py</code> retains target-bin predictions, production-workspace B spectra, and cross terms.
    <code>analysis/check_section5_geometry.py</code> and <code>analysis/check_beam_inputs.py</code> carry the verification gates.
    This thread does not edit <code>paper/main.tex</code>.</p></body></html>'''
    DEST.write_text(page)
    print(DEST)


if __name__=='__main__':
    run()
