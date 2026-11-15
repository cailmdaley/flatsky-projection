"""Bandpowers of the projection suite, and the theory they are compared to.

Every analysis step reads its upstream outputs through a `Run`, which holds
the directories it was handed on the command line (`add_inputs` declares the
flags, one per output id) and nothing else. The binning, number of
realizations and pixel-window treatment come from the bandpowers' own index
files; no decision value is typed here.

The production run
------------------
`bandpowers/bandpowers_{proj}_seed{s:04d}.npz`, one file per (seed,
projection) holding every footprint by name -- the seven RA/dec rectangles of
`src/stages/_footprints.py`, SPT's 1500d field among them. The paper studies four projections,
`paper_style.PROJS`, and every array here is indexed in that order; the run
also holds a fifth, ProjSIN, which nothing in `analysis/` reads. What it did:

  - Input: HEALPix *unlensed* TQU realizations of the Planck 2018 best-fit
    cosmology, so the true B is exactly zero and every recovered B bandpower
    is leakage.
  - map2alm to the band limit, monopole and dipole filtered. Sky power
    therefore exists to l = lmax, and any forward model compared to these
    spectra must be band-limited there.
  - Synthesized to a flat grid by bilinear interpolation with oversampling,
    centred on the patch centre of `_footprints`.
  - spt3g maps.FlattenPol applied: the projection-induced spin-2 rotation is
    undone before spectra are taken, leaving only the symmetric (shear +
    magnification) part of the Jacobian.
  - Two legs per realization. Leg 0 is QU -- project Q and U, flatten, then
    decompose to E/B on the plane, which is what a real analysis does and what
    carries the leakage. Leg 1 is EB, decomposed on the sphere first.
  - NaMaster NmtFieldFlat with B purification on each apodized footprint.

Pixel window
------------
The run divides out the flat-sky top-hat of the pixel and nothing else
(`pixel_window: flat_only`, recorded in every unit's index file, which `Run`
checks). The inputs carry no HEALPix window, so there is none to divide.
Loading additionally removes the coherent ring-interpolation and finite-rebin
windows, the `simulation_windows` output (`analysis/simulation_windows.py`).
These are integrated with the known input spectrum and the full local
Jacobian separately for each projection, footprint and spin block. Pass
`correct_windows=False` to recover the production files unchanged.

This removes only windows introduced by the simulation, not the geometric
projection effect. It does not remove aliased power or calibrate the STG BB
estimator floor. See `pipeline_windows.py`.
"""

import json
import sys
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import PROJ_LABEL, PROJS  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "stages"))
from _footprints import FIELD, FOOTPRINTS, SWEEP  # noqa: E402,F401
from _footprints import FSKY_PERCENT as _FSKY_PERCENT  # noqa: E402

PROJ_KIND = {
    "ProjSFL": "pseudocylindrical, equal-area",
    "ProjSTG": "conformal",
    "ProjZEA": "zenithal equal-area",
    "ProjBICEP": "cylindrical",
}
# The sweep's unapodized sky fractions, in percent, in registry order.
FSKY_PERCENT = np.array(_FSKY_PERCENT)

QU_LEG = 0
EB_LEG = 1


def add_inputs(parser, *names):
    """Declare one required `--<output-id>` path flag per upstream output."""
    for name in names:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)


def _index(directory):
    """(lmax, delta_l, seeds) of a bandpower directory, from its index files.

    Refuses bandpowers made with any pixel-window treatment but flat_only:
    the simulation windows are relative to the flat top-hat alone.
    """
    idx = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("index_*.json"))]
    (lmax,), (delta_l,) = {i["lmax"] for i in idx}, {i["delta_ell"] for i in idx}
    windows = {i["pixel_window"] for i in idx}
    if windows != {"flat_only"}:
        raise ValueError(f"{directory}: pixel_window {sorted(windows)}, "
                         "expected only 'flat_only'")
    seeds = sorted({s for i in idx for s in i["seeds"]})
    if seeds != list(range(len(seeds))):
        raise ValueError(f"{directory}: seeds {seeds} are not 0..n-1")
    return lmax, delta_l, seeds


def _rows(z, footprints):
    """Positions of `footprints` along a file's footprint axis."""
    names = list(z["footprints"])
    return [names.index(name) for name in footprints]


@dataclass(frozen=True)
class Run:
    """The upstream outputs one analysis step reads, by output id.

    Each attribute is the output's directory, or None when the step does not
    read it. The binning comes from `bandpowers`, or `scalar_bandpowers` when only the scalar leg is read.
    """
    bandpowers: Path | None = None
    scalar_bandpowers: Path | None = None
    simulation_windows: Path | None = None
    theory_spectra: Path | None = None
    shear_moments: Path | None = None

    @classmethod
    def from_args(cls, args):
        """The `Run` of a parsed command line, from the flags `add_inputs` declared."""
        return cls(**{f.name: getattr(args, f.name, None) for f in fields(cls)})

    # ---- binning ------------------------------------------------------------

    @cached_property
    def _meta(self):
        return _index(self.bandpowers or self.scalar_bandpowers)

    @property
    def lmax(self):
        return self._meta[0]

    @property
    def delta_l(self):
        return self._meta[1]

    @property
    def nsims(self):
        return len(self._meta[2])

    @property
    def nbins(self):
        return self.lmax // self.delta_l

    @property
    def bin_lo(self):
        return np.arange(self.nbins) * self.delta_l

    @property
    def bin_hi(self):
        return (np.arange(self.nbins) + 1) * self.delta_l

    @property
    def ell(self):
        """Bin midpoints, which are NmtBinFlat's effective multipoles here."""
        return 0.5 * (self.bin_lo + self.bin_hi).astype(float)

    def ell_eff(self):
        """The NmtBinFlat effective multipoles as written by the run."""
        with np.load(self.bandpowers / f"bandpowers_{PROJS[0]}_seed0000.npz") as z:
            return z["ell_eff"]

    # ---- bandpowers ---------------------------------------------------------

    def _load(self, footprints, spec, leg):
        """(n_footprints, n_projs, nsims, nbins) bandpowers of one leg."""
        key = spec.lower()
        if key not in ("ee", "bb"):
            raise ValueError(f"spec must be EE or BB, got {spec!r}")
        out = np.full((len(footprints), len(PROJS), self.nsims, self.nbins), np.nan)
        for j, proj in enumerate(PROJS):
            for s in range(self.nsims):
                path = self.bandpowers / f"bandpowers_{proj}_seed{s:04d}.npz"
                with np.load(path) as z:
                    arr = z[key][leg, 0, :, 0, :][_rows(z, footprints)]
                if arr.shape != (len(footprints), self.nbins):
                    raise ValueError(f"{path.name}: {arr.shape}")
                out[:, j, s, :] = arr
        return out

    def load(self, spec="EE", leg=QU_LEG, correct_windows=True):
        """Sweep bandpowers as (n_footprints, n_projs, nsims, nbins), QU leg by default.

        The footprints come in registry order, matching `FSKY_PERCENT`. `leg=EB_LEG` reads
        the E/B leg instead, whose "EE" and "BB" are the spin-0 auto-spectra
        of the sphere-decomposed E and B maps. The flat-sky top-hat is
        already divided out. By default the remaining known simulation
        windows are removed too (see the module docstring).
        """
        out = self._load(SWEEP, spec, leg)
        if correct_windows:
            out /= self.simulation_window(spec, leg=leg)[:, :, None, :]
        return out

    def load_field(self, spec="EE", leg=QU_LEG, correct_windows=True):
        """1500d-field bandpowers as (n_projs, nsims, nbins), QU leg by default."""
        out = self._load([FIELD], spec, leg)[0]
        if correct_windows:
            out /= self.simulation_window(spec, field=True, leg=leg)[:, None, :]
        return out

    def load_scalar(self, spec="TT", field=False, correct_windows=True):
        """Scalar bandpowers, (footprint, projection, seed, band) over the sweep, or the field alone.

        `spec` is TT or white. Both use the EE bins and flat_only window
        correction. White skies have unit input C_l from ell=2 through lmax.
        By default their own effective simulation window is removed, not the
        CMB-calibrated one.
        """
        kind = {"tt": "tt", "white": "white"}[spec.lower()]
        footprints = [FIELD] if field else SWEEP
        directory = self.scalar_bandpowers
        _index(directory)
        out = np.empty((len(footprints), len(PROJS), self.nsims, self.nbins))
        for j, proj in enumerate(PROJS):
            for seed in range(self.nsims):
                with np.load(directory / f"bandpowers_{kind}_{proj}_seed{seed:04d}.npz") as z:
                    if not np.array_equal(z["ell_eff"], self.ell):
                        raise ValueError(f"{directory}: scalar bins differ from EE")
                    out[:, j, seed] = z["cl"][_rows(z, footprints)]
        out = out[0] if field else out
        if correct_windows:
            window = self.simulation_window(spec, field=field)
            out /= window[:, None, :] if field else window[:, :, None, :]
        return out

    def simulation_window(self, spec="EE", field=False, leg=QU_LEG):
        """Effective window for the known simulation input, not an arbitrary sky.

        Ring interpolation in the sky frame and the exact finite-subsample grid
        average, relative to the continuous top-hat already divided out in
        production. Shape (n_projs, nbins) on the field, (n_footprints, n_projs,
        nbins) on the sweep.
        """
        with np.load(self.simulation_windows / "simulation_windows.npz") as z:
            if not np.array_equal(z["ell"], self.ell):
                raise ValueError(f"{self.simulation_windows}: bins differ from the bandpowers'")
            return z[window_key(spec, field, leg)]

    # ---- theory -------------------------------------------------------------

    def bin_theory(self, cl):
        """Average an unbinned C_l over each bandpower.

        Binning matters: the acoustic peaks move fast compared to a bandpower,
        so evaluating theory at bin centres instead of averaging over the bin
        manufactures percent-level wiggles that look like a projection effect.
        """
        return np.array([np.nanmean(cl[lo:hi]) for lo, hi in zip(self.bin_lo, self.bin_hi)])

    def theory(self, spec="unlensed_EE"):
        """A binned CAMB spectrum in the sims' cosmology, on the production bins.

        `spec` names a key of theory_spectra.npz: unlensed_ or lensed_ crossed
        with TT, EE, BB, TE. The sims are unlensed, so unlensed_EE is the input
        the projected EE must be compared against; lensed_BB is the floor any
        claimed leakage has to be stated relative to.
        """
        with np.load(self.theory_spectra / "theory_spectra.npz") as data:
            ell = data["ell"]
            cl = data[spec]
        full = np.zeros(int(ell.max()) + 1)
        full[ell] = cl
        full[0] = 0.0
        return self.bin_theory(full)

    # ---- geometry -----------------------------------------------------------

    def moments(self, footprints=SWEEP):
        """Shear moments of `footprints`, stacked (footprint, projection, ...).

        The `shear_moments` output (`analysis/shear_moments.py`), evaluated on
        the pipeline's own masks. The default, the seven footprints in registry
        order, matches `FSKY_PERCENT`, so mask index i is a sweep footprint throughout.
        """
        path = self.shear_moments / "shear_moments.npz"
        with np.load(path) as f:
            if list(f["projs"]) != PROJS:
                raise ValueError(f"{path}: projections {list(f['projs'])}, expected {PROJS}")
            rows = [list(f["footprints"]).index(name) for name in footprints]
            return {k: f[k][rows] for k in f.files
                    if k not in ("footprints", "projs", "n_bins")}

    def field_moments(self):
        """`moments` of SPT's 1500d field, shaped like one mask (i = 0)."""
        return self.moments([FIELD])


def window_key(spec, field, leg):
    """The array name of one window in `simulation_windows.npz`."""
    return f"{spec.upper()}_{'field' if field else 'sweep'}_{('qu', 'eb')[leg]}"


def flat_pixwin(ell, res_arcmin, n_phi=512):
    """Azimuthally averaged square-pixel window, in C_l (i.e. already squared).

    Rebinning an oversampled grid down to `res_arcmin` pixels is a top-hat
    average, whose 2D transform is sinc(lx d/2) sinc(ly d/2). NaMaster bins in
    |l| annuli, so average |W|^2 around each annulus.
    """
    d = np.deg2rad(res_arcmin / 60.0)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    lx = np.outer(ell, np.cos(phi))
    ly = np.outer(ell, np.sin(phi))
    w = np.sinc(lx * d / (2 * np.pi)) * np.sinc(ly * d / (2 * np.pi))
    return (w ** 2).mean(axis=1)


def rebin_pseudo(pseudo, counts, edges):
    """Rebin unit-width pseudo-C_ell with Fourier-cell weighting.

    Integer edges select unions of the stored [ell, ell+1) annuli.
    Empty fine annuli contribute zero weight; an empty target bin raises.
    Averaging decoupled bandpowers or averaging pseudo-spectra without these
    counts does not implement the NaMaster estimator.
    """
    edges = np.asarray(edges)
    if (edges.ndim != 1 or len(edges) < 2 or
            not np.all(edges == edges.astype(int)) or
            np.any(np.diff(edges) <= 0) or edges[0] < 0 or edges[-1] > len(counts)):
        raise ValueError('edges must be increasing integers within the stored ell range')
    edges = edges.astype(int)
    weights = np.diff(np.r_[0, np.cumsum(counts)][edges])
    if np.any(weights == 0):
        raise ValueError('target binning contains empty Fourier annuli')
    return np.stack([(pseudo[..., lo:hi] * counts[lo:hi]).sum(axis=-1) / weight
                     for lo, hi, weight in zip(edges[:-1], edges[1:], weights)], axis=-1)


def decouple_fine(path, delta_ell, *, workspace=None, masks_dir=None, cache_dir=None):
    """Return TT, EE, BB bandpowers from a saved finest pseudo-spectrum.

    ``delta_ell`` is a positive integer; the last bin ends at stored lmax.
    Supply ``workspace`` as a directory of w00/w22 FITS files at this width
    or a finer width whose bins partition these bins exactly, or
    ``masks_dir`` and ``cache_dir`` to build and cache a workspace for this
    width. Workspace construction can take hours and should run on the grid.
    The flat pixel window is divided out after decoupling, as in production.
    Only the QU leg is stored by the finest pseudo-spectrum stage.
    """
    import pymaster as nmt
    from namaster_spectra import compute_pixel_window, load_mask
    from coupling_matrices import atomic_write_workspace

    if int(delta_ell) != delta_ell or delta_ell < 1:
        raise ValueError('delta_ell must be a positive integer')
    with np.load(path) as z:
        counts, pol, tt = z['counts'], z['pol'], z['tt']
        proj, footprint = str(z['projection']), str(z['footprint'])
        purify_b = bool(z['purify_b'])
        geometry = tuple(float(z[k]) for k in ('nx', 'ny', 'lx_rad', 'ly_rad'))
        mask_hash = str(z['mask_sha256'])
    lmax = len(counts)
    edges = np.unique(np.r_[np.arange(0, lmax, int(delta_ell)), lmax])
    pseudo_pol = rebin_pseudo(pol, counts, edges)
    pseudo_tt = rebin_pseudo(tt, counts, edges)
    bins = nmt.NmtBinFlat(edges[:-1].astype(float), edges[1:].astype(float))
    target = (Path(workspace) if workspace is not None else
              Path(cache_dir) / f'd{int(delta_ell)}' / proj / mask_hash)
    target.mkdir(parents=True, exist_ok=True)
    result = {}
    for spin, pseudo in [(0, pseudo_tt), (2, pseudo_pol)]:
        ws_path = target / f'w{spin}{spin}_{proj}_{footprint}.fits'
        w = nmt.NmtWorkspaceFlat()
        if ws_path.exists():
            w.read_from(str(ws_path))
        elif masks_dir is not None:
            mask, _, lx, ly = load_mask(Path(masks_dir), proj, footprint)
            f = nmt.NmtFieldFlat(lx, ly, mask, None, spin=spin,
                                 purify_b=purify_b if spin == 2 else False)
            w.compute_coupling_matrix(f, f, bins)
            import hashlib
            if hashlib.sha256(mask.tobytes()).hexdigest() != mask_hash:
                raise ValueError('masks_dir mask differs from the stored pseudo-spectrum mask')
            atomic_write_workspace(w, ws_path)
            from astropy.io import fits
            with fits.open(ws_path, mode='update') as hdus:
                hdus[0].header['MASKHASH'] = mask_hash
        else:
            raise FileNotFoundError(f'{ws_path}; supply matching workspaces or masks_dir')
        # Flat workspaces interleave spectra within each bin in their FITS MCM.
        from astropy.io import fits
        with fits.open(ws_path) as hdus:
            stored_geometry = tuple(hdus['FS_INFO'].header[k] for k in ('NX', 'NY', 'LX', 'LY'))
            if not np.allclose(geometry, stored_geometry, rtol=1e-12, atol=0):
                raise ValueError(f'{ws_path}: workspace geometry differs from pseudo-spectrum')
            if hdus[0].header.get('MASKHASH', mask_hash) != mask_hash:
                raise ValueError(f'{ws_path}: workspace mask differs from pseudo-spectrum')
            stored_edges = hdus['BINS_SUMMARY'].data
            source_edges = np.r_[stored_edges.field(0), stored_edges.field(1)[-1]]
            if spin == 2 and bool(hdus[0].header['PURE_B1']) != purify_b:
                raise ValueError(f'{ws_path}: workspace purification differs from pseudo-spectrum')
            same_bins = np.array_equal(source_edges, edges)
            if not same_bins:
                matrix = rebin_flat_response(hdus['MCM_BINNED'].data, source_edges,
                                             edges, counts, 1 if spin == 0 else 4)
        if same_bins:
            cl = w.decouple_cell(pseudo)
        else:
            from scipy.linalg import solve
            cl = solve(matrix, pseudo.T.reshape(-1)).reshape(-1, pseudo.shape[0]).T
        if not np.isfinite(cl).all():
            raise ValueError(f'{ws_path}: nonfinite decoupled spectrum')
        if spin == 0:
            result['tt'] = cl[0]
        else:
            result.update(ee=cl[0], bb=cl[3])
    ell = bins.get_effective_ells()
    window = compute_pixel_window('flat_only', bins, ell, lmax)
    return dict(ell_eff=ell, **{key: value / window for key, value in result.items()})


def rebin_flat_response(matrix, source_edges, target_edges, counts, ncls):
    """Rebin a NaMaster flat-sky response as P M Q.

    Rows and columns interleave spectra inside each bin in NaMaster FITS files.
    P averages output pseudo-spectra by Fourier-cell count; Q holds each input
    bandpower constant across its constituent source bins.
    Target edges must be a subset of source edges with the same endpoints.
    """
    source_edges, target_edges = np.asarray(source_edges), np.asarray(target_edges)
    if (source_edges[0] != target_edges[0] or source_edges[-1] != target_edges[-1]
            or not np.isin(target_edges, source_edges).all()):
        raise ValueError('target bins must be unions of source bins over the same range')
    pos = np.searchsorted(source_edges, target_edges)
    fine_counts = np.diff(np.r_[0, np.cumsum(counts)][source_edges.astype(int)])
    nsource = len(source_edges) - 1
    rows = matrix.reshape(nsource, ncls, nsource, ncls)
    # Sum source columns: a constant target C_l fills each input source bin.
    columns = np.stack([rows[:, :, lo:hi, :].sum(axis=2)
                        for lo, hi in zip(pos[:-1], pos[1:])], axis=2)
    out = np.stack([(columns[lo:hi] * fine_counts[lo:hi, None, None, None]).sum(axis=0)
                    / fine_counts[lo:hi].sum()
                    for lo, hi in zip(pos[:-1], pos[1:])])
    return out.reshape((len(target_edges) - 1) * ncls, -1)
