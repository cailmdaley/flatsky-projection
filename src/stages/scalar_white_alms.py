"""Independent Gaussian scalar skies with C_l=1 for 2 <= l <= lmax."""
import argparse
from pathlib import Path
from _decisions import add_common_args, parse_lmax, resolve_seeds, set_threads, write_index, index_name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lmax', required=True, type=parse_lmax)
    add_common_args(p, seeds=True)
    args = p.parse_args()
    set_threads(args.max_procs)
    import numpy as np
    import healpy as hp
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = resolve_seeds(args)
    cl = np.ones(args.lmax + 1)
    cl[:2] = 0
    for seed in seeds:
        # A separate namespace from the external CMB simulation seed sequence.
        rng_seed = int(np.random.SeedSequence([20261105, seed]).generate_state(1)[0])
        np.random.seed(rng_seed)
        alm = hp.synalm(cl, lmax=args.lmax, new=True)
        path = out / f'alms_seed{seed:04d}.fits'
        tmp = out / f'alms_seed{seed:04d}.tmp.fits'
        hp.write_alm(str(tmp), alm, overwrite=True)
        tmp.replace(path)
    write_index(out, {'seeds': seeds, 'lmax': args.lmax, 'cl': 1.0,
                      'ell_min': 2, 'seed_namespace': 20261105},
                name=index_name(seeds))


if __name__ == '__main__':
    main()
