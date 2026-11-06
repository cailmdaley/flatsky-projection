"""Check canonical residual controls, every unit hash and explicit source bytes.

Run with the installed lightcone Python environment after `lc verify -u highell`.
This supplements lc verify, which checks canonical output hashes and upstream
manifest versions but does not recursively recompute the gathered unit hashes.
No manifest is written or modified here.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import yaml
from lightcone.engine.manifest import read_manifest, sha256_dir

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/figures/residual_anomalies/provenance.json'


def main():
    spec = yaml.safe_load((ROOT/'astra.yaml').read_text())
    sources = {i['id']: i['source'] for i in spec['inputs'] if 'source' in i}
    rows = []
    for name, count in [('residual_pure_e_test', 16), ('residual_map_upsample', 16),
                        ('residual_ring_test', 4)]:
        directory = ROOT/'results/highell'/name
        manifest = read_manifest(directory)
        assert manifest is not None, name
        assert sha256_dir(directory) == manifest['data_version'], name
        assert len(manifest['units']) == count, name
        for input_id, version in manifest['input_versions'].items():
            if input_id not in sources:
                upstream = read_manifest(ROOT/'results/highell'/input_id)
                assert upstream is not None and upstream['data_version'] == version, (name, input_id)
        dependencies = ['src/stages/_decisions.py', 'src/stages/namaster_spectra.py']
        if name == 'residual_ring_test':
            dependencies.append('src/stages/bin_covariance_sims.py')
        checked = set(dependencies)
        for unit, fingerprint in manifest['units'].items():
            unit_dir = directory.with_name(name+'__units')/unit
            unit_manifest = read_manifest(unit_dir)
            assert unit_manifest is not None, (name, unit)
            assert sha256_dir(unit_dir) == fingerprint == unit_manifest['data_version'], (name, unit)
            # External files are mtime/size tracked by this lc version.
            # Verify their content independently against the unit's git pin.
            paths = dependencies+[sources[source_id] for source_id in unit_manifest['input_versions']
                                  if source_id.startswith('residual_') and source_id in sources]
            for source in paths:
                disk = (ROOT/source).read_bytes()
                pinned = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                                   unit_manifest['git_sha']+':'+source])
                assert disk == pinned, (name, unit, source)
                checked.add(source)
        row = dict(output=name, nunit=count, data_version=manifest['data_version'],
                   sources={path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                            for path in sorted(checked)})
        rows.append(row)
        print(row, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
