"""Exercise the real CLI, check output validity, determinism and fallback inputs."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import main as model

ROOT = Path(__file__).resolve().parent
KIT = Path('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/x/gta6-twitch-ccu-forecast-01-kit/input')


def validate(path):
    assert path.stat().st_size <= 65536
    output = json.loads(path.read_bytes())
    assert set(output) == {'hours'}
    rows = output['hours']
    assert len(rows) == 23
    assert sorted(row['hour'] for row in rows) == list(range(2, 25))
    for row in rows:
        assert set(row) == {'hour', 'q'}
        q = row['q']
        assert len(q) == 5
        assert all(type(x) in (int, float) and math.isfinite(x) and 0 < x <= 1e9 for x in q)
        assert q == sorted(q)
    return output


def run(source, output):
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(source), MAKEFUN_OUTPUT_DIR=str(output))
    start = time.perf_counter()
    subprocess.run([sys.executable, '-P', '-s', str(ROOT / 'main.py')], env=env, check=True)
    return time.perf_counter() - start


def main():
    output = ROOT / 'out'
    elapsed = run(KIT, output)
    first = {p.name: p.read_bytes() for p in sorted(output.glob('*.out'))}
    assert set(first) == {'p00.out', 'p01.out', 'p02.out', 'p03.out'}
    elapsed_second = run(KIT, output)
    assert first == {p.name: p.read_bytes() for p in sorted(output.glob('*.out'))}
    print(f'Two complete CLI runs: {elapsed:.4f}s, {elapsed_second:.4f}s; byte-identical outputs.')
    with tempfile.TemporaryDirectory(prefix='validation-', dir=ROOT) as tmp:
        tmp = Path(tmp)
        source, destination = tmp / 'input', tmp / 'output'
        source.mkdir()
        for path in sorted(KIT.glob('*.in')):
            link = source / path.name
            link.symlink_to(path)
            seconds = run(source, destination)
            assert seconds < 8
            result = destination / (path.stem + '.out')
            validate(result)
            assert result.read_bytes() == first[result.name]
            print(f'{path.stem}: valid, {result.stat().st_size} bytes, {seconds:.4f}s, SHA256 {hashlib.sha256(result.read_bytes()).hexdigest()}')
            link.unlink()
        for p in sorted(output.glob('*.out')):
            validate(p)
        # Real malformed JSON and unsafe ids must still emit valid fallback files.
        (source / 'broken.in').write_text('{not json')
        (source / 'unsafe.in').write_text(json.dumps({'id': '../bad', 'known': [{'hour': 1, 'viewers': 1234}]}))
        run(source, destination)
        validate(destination / 'broken.out')
        validate(destination / 'unsafe.out')
        for case in (None, {}, [], {'known': None}, {'day_start': 'bad'},
                     {'known': [{'hour': 1, 'viewers': float('nan')}]},
                     {'known': [{'hour': 1, 'viewers': 1e-300}]},
                     {'known': [{'hour': 1, 'viewers': 1e300}]}):
            payload = model.forecast_case(case)
            check = destination / 'fallback.out'
            check.write_text(json.dumps(payload, allow_nan=False))
            validate(check)
        print('Malformed input, missing features, unsafe id, and extreme-count fallback checks passed.')


if __name__ == '__main__':
    main()
