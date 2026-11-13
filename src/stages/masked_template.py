"""Masked noisy pure-E template comparison on the SFL 1500d field, one seed."""
import argparse
from pathlib import Path
import sys
from _decisions import parse_delta_ell, parse_lmax, parse_purify_b, set_threads


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps',required=True,help='projected_qu output directory')
    p.add_argument('--masks',required=True,help='flatsky_masks output directory')
    p.add_argument('--theory',required=True,help='theory_spectra npz')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--lmax',type=parse_lmax,required=True)
    p.add_argument('--delta-ell',type=parse_delta_ell,required=True)
    p.add_argument('--purify-b',type=parse_purify_b,required=True)
    p.add_argument('--noise',type=float,required=True,help='white Q/U depth per split, uK-arcmin')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    set_threads(4)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis'))
    from masked_template import run
    run(a.maps,a.masks,a.theory,a.seed,a.out,a.lmax,a.delta_ell,a.purify_b,a.noise)


if __name__=='__main__': main()
