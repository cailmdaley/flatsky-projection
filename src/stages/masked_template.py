"""Masked/noisy E-template control on the SFL 1500d field."""
import argparse
from pathlib import Path
import sys
from _decisions import set_threads


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--maps',required=True)
    p.add_argument('--masks',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    set_threads(4)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis'))
    from masked_template import run
    run(a.maps,a.masks,a.seed,a.out)


if __name__=='__main__': main()
