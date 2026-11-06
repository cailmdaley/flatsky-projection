"""Run the same lognormal alms with finer sphere and/or chart sampling."""
import argparse
import json
from pathlib import Path
from beyond_two_point import run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--resolution', choices=('sphere', 'plane', 'both'), required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
nside = 1024 if a.resolution in ('sphere', 'both') else 512
pixel = .05 if a.resolution in ('plane', 'both') else .1
result = run(a.seed, nside, pixel)
a.out.mkdir(parents=True, exist_ok=True)
name = f'seed{a.seed:03d}_{a.resolution}.json'
(a.out/name).write_text(json.dumps(result, indent=2)+'\n')
print(name, flush=True)
