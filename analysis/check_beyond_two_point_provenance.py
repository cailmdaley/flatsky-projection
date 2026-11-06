"""Verify just the new outputs, all unit hashes, and their actual source pins.

Run with lightcone's Python (same environment as the lc executable):
  ~/.local/share/uv/tools/lightcone-cli/bin/python analysis/check_beyond_two_point_provenance.py

lc verify has no output selector. Project its unchanged declarations onto an
isolated temporary spec rather than rehashing unrelated terabyte map stages.
This changes neither production specs, manifests nor data.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml
from lightcone.engine.manifest import read_manifest, sha256_dir

ROOT = Path(__file__).resolve().parents[1]
IDS = ('beyond_two_point', 'beyond_two_point_resolution')
spec = yaml.safe_load((ROOT/'astra.yaml').read_text())
spec['outputs'] = [o for o in spec['outputs'] if o['id'] in IDS]
assert len(spec['outputs']) == 2
for output in IDS:
    directory = ROOT/'results/highell'/output
    manifest = read_manifest(directory)
    assert manifest['data_version'] == sha256_dir(directory)
    assert len(manifest['units']) == (24 if output == IDS[0] else 12)
    sources = ['src/stages/beyond_two_point.py']
    if output == IDS[1]:
        sources.append('src/stages/beyond_two_point_resolution.py')
    for unit, version in manifest['units'].items():
        path = ROOT/'results/highell'/(output+'__units')/unit
        um = read_manifest(path)
        assert um['data_version'] == version == sha256_dir(path)
        for source in sources:
            pinned = subprocess.check_output(['git', 'show', f'{um["git_sha"]}:{source}'], cwd=ROOT)
            assert pinned == (ROOT/source).read_bytes(), (unit, source)
        for data in path.glob('seed*.json'):
            assert data.read_bytes() == (directory/data.name).read_bytes()
    print(output, 'canonical data, every unit, gathered bytes and source pins match', flush=True)
with tempfile.TemporaryDirectory(prefix='beyond-two-point-verify-', dir=os.environ.get('TMPDIR')) as temporary:
    tmp = Path(temporary)
    (tmp/'astra.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
    (tmp/'results').symlink_to((ROOT/'results').resolve())
    subprocess.run(['lc', 'verify', '-u', 'highell'], cwd=tmp, check=True)
