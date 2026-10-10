"""Check a source-only release for excluded binary/data artifacts and syntax."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BANNED_SUFFIXES = {'.pt', '.pth', '.torch', '.ckpt', '.safetensors', '.onnx',
                   '.h5', '.hdf5', '.npy', '.npz', '.pkl', '.pickle', '.bin',
                   '.whl', '.mp4', '.avi', '.pdf', '.docx', '.zip', '.pyc'}
BANNED_DIRS = {'data', 'assets', 'weights', 'checkpoints', 'model_ckpt',
               'outputs', 'experiments', 'log', 'logs', 'wandb'}


def main():
    issues = []
    count = 0
    size = 0
    for file in ROOT.rglob('*'):
        if not file.is_file():
            continue
        rel = file.relative_to(ROOT)
        if any(part in {'.git', '__pycache__', '.pytest_cache'} or part.endswith('.egg-info') for part in rel.parts):
            continue
        count += 1
        size += file.stat().st_size
        if file.suffix.lower() in BANNED_SUFFIXES or any(part in BANNED_DIRS for part in rel.parts):
            issues.append(f'excluded artifact: {rel.as_posix()}')
        if file.suffix == '.py':
            try:
                ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(rel))
            except (SyntaxError, UnicodeError) as error:
                issues.append(f'invalid source: {rel}: {error}')
        if not file.stat().st_size and file.name != '__init__.py':
            issues.append(f'empty file: {rel.as_posix()}')
    if issues:
        print('\n'.join(issues))
        return 1
    print(f'PASS: {count} source/config/document files, {size:,} bytes; no data or model artifacts.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
