"""Build the Codabench model-submission ZIP from submission/ (model.py at the ZIP root).

  uv run python scripts/make_zip.py [--out dist/FAIR-Universe-2026-v3-16b.zip]

Every file is first checked against SUBMISSION_SHA256SUMS, so the ZIP holds exactly those bytes. They are the files
of the package scored on the public leaderboard; four docstrings differ only in a source path. Entries are
sorted and carry a fixed timestamp, so the ZIP itself is reproducible.
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'dist' / 'FAIR-Universe-2026-v3-16b.zip'))
    args = ap.parse_args()
    expected = {}
    for line in (ROOT / 'SUBMISSION_SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    src = ROOT / 'submission'
    found = sorted(p.name for p in src.iterdir() if p.is_file())
    if found != sorted(expected):
        raise SystemExit(f'submission/ files differ from SUBMISSION_SHA256SUMS: '
                         f'extra {sorted(set(found) - set(expected))} missing {sorted(set(expected) - set(found))}')
    for name, digest in expected.items():
        if hashlib.sha256((src / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f'{name}: sha256 mismatch')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(expected):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (src / name).read_bytes())
    print(f'{out}  {len(expected)} files  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
