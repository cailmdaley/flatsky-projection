#!/usr/bin/env python3
"""Paired finer-HEALPix CMB backgrounds; compact source/white noise unchanged."""
import argparse
from pathlib import Path
from local_source_sims import run

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--nside',type=int,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
run(a.seed,a.out,a.nside)
(a.out/f'sources_{a.seed:03d}.json').rename(a.out/f'resolution_{a.nside}_{a.seed:03d}.json')
if a.seed==0:
    (a.out/'source_examples.npz').rename(a.out/f'examples_{a.nside}.npz')
