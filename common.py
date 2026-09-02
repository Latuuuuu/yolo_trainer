"""Shared helpers for train.py / predict.py / export.py.

Keeps the three entry points free of path juggling and gives beginners an
actionable message instead of a stack trace.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / 'datasets'
RUNS_DIR = REPO_ROOT / 'runs'
WEIGHTS_DIR = REPO_ROOT / 'weights'

DOCS_HINT = 'See the Troubleshooting section in README.md'


def die(message, *hints):
    """Print a readable error and exit. Never raises."""
    print(f'\n[error] {message}', file=sys.stderr)
    for hint in hints:
        print(f'        {hint}', file=sys.stderr)
    print('', file=sys.stderr)
    sys.exit(1)


def resolve(path, base=REPO_ROOT):
    """Make a config path absolute, relative to the repository root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def list_datasets():
    """Dataset folders that contain a data.yaml, for error messages."""
    if not DATASETS_DIR.is_dir():
        return []
    return sorted(
        d.name for d in DATASETS_DIR.iterdir()
        if d.is_dir() and (d / 'data.yaml').is_file()
    )


def check_dataset(data_path):
    """Validate a data.yaml before handing it to Ultralytics."""
    import yaml

    path = resolve(data_path)
    if not path.is_file():
        available = list_datasets()
        hints = [f'Looked for: {path}']
        if available:
            hints.append('Datasets available in datasets/:')
            hints += [f'  - datasets/{name}/data.yaml' for name in available]
        else:
            hints.append('No dataset with a data.yaml found under datasets/.')
            hints.append('Unzip your dataset (YOLO format) into datasets/ first.')
        die(f'Dataset config not found: {data_path}', *hints)

    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        die(f'{path} is not valid YAML.', str(e))

    missing = [k for k in ('train', 'val', 'names') if k not in cfg]
    if missing:
        die(
            f'{path} is missing required key(s): {", ".join(missing)}',
            'A data.yaml needs at least:',
            '  train: train/images',
            '  val: valid/images',
            '  names:',
            '    0: my_class',
        )

    # Warn (do not fail) when the image folders are missing: Ultralytics also
    # supports `path:` prefixes and file lists, so we cannot be certain here.
    root = resolve(cfg['path'], path.parent) if cfg.get('path') else path.parent
    for key in ('train', 'val'):
        value = cfg[key]
        if isinstance(value, str) and not resolve(value, root).exists():
            print(f'[warn] {key}: {value} does not exist under {root}')

    return path, cfg


def check_cuda(device):
    """Fail early with a clear message when the GPU is not usable."""
    if str(device).lower() == 'cpu':
        return
    try:
        import torch
    except ImportError:
        die('PyTorch is not installed.', 'Run everything inside the container (see README).')

    if not torch.cuda.is_available():
        die(
            'No CUDA GPU is visible to PyTorch — this project needs one.',
            'Checklist:',
            '  1. nvidia-smi works on the host',
            '  2. nvidia-container-toolkit is installed and Docker was restarted',
            '  3. you are inside the container started by setup.sh',
            '  4. or pass device=cpu to run (very slowly) on CPU',
            DOCS_HINT,
        )
