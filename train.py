"""Train a YOLO model from a YAML config.

    python train.py --config configs/example_detect.yaml
    python train.py --config configs/example_obb.yaml --set epochs=50 --name quick_test

Everything you normally tune lives in configs/ — you should not need to edit
this file. Any Ultralytics training argument can be set in the config or with
--set, see https://docs.ultralytics.com/modes/train/#train-settings
"""

import argparse

import yaml

from common import REPO_ROOT, check_cuda, check_dataset, die, resolve

DEFAULT_CONFIG = 'configs/default.yaml'


def parse_args():
    p = argparse.ArgumentParser(
        description='Train a YOLO model (config driven).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--config', default=DEFAULT_CONFIG, help=f'training config (default: {DEFAULT_CONFIG})')
    p.add_argument('--model', help='override the config: weights .pt or architecture .yaml')
    p.add_argument('--data', help='override the config: path to data.yaml')
    p.add_argument('--name', help='override the config: run name (results land in runs/<name>)')
    p.add_argument('--device', help='override the config: GPU index, "0,1", or cpu')
    p.add_argument('--epochs', type=int, help='override the config: number of epochs')
    p.add_argument('--batch', type=int, help='override the config: batch size')
    p.add_argument('--imgsz', type=int, help='override the config: image size')
    p.add_argument('--resume', nargs='?', const=True, metavar='CHECKPOINT',
                   help='resume training: no value = runs/<name>/weights/last.pt, or give a path')
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help='override any Ultralytics argument, repeatable (e.g. --set lr0=0.001)')
    return p.parse_args()


def load_config(config_path):
    path = resolve(config_path)
    if not path.is_file():
        available = sorted(p.name for p in (REPO_ROOT / 'configs').glob('*.yaml'))
        die(f'Config not found: {path}',
            'Available configs: ' + (', '.join(f'configs/{n}' for n in available) or '(none)'))
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        die(f'{path} is not valid YAML.', str(e))

    if 'model' not in cfg or 'data' not in cfg:
        die(f'{path} must define both `model:` and `data:`.',
            f'Compare with {DEFAULT_CONFIG}.')
    args = cfg.get('args') or {}
    if not isinstance(args, dict):
        die(f'{path}: `args:` must be a mapping of Ultralytics arguments.')
    return cfg, dict(args)


def apply_overrides(cfg, args, cli):
    """CLI beats config. --set accepts any Ultralytics argument."""
    for key in ('model', 'data'):
        if getattr(cli, key):
            cfg[key] = getattr(cli, key)
    for key in ('name', 'device', 'epochs', 'batch', 'imgsz'):
        if getattr(cli, key) is not None:
            args[key] = getattr(cli, key)
    for item in cli.overrides:
        if '=' not in item:
            die(f'--set expects KEY=VALUE, got: {item}', 'Example: --set lr0=0.001')
        key, _, raw = item.partition('=')
        # yaml.safe_load gives us ints, floats, bools and lists for free
        args[key.strip()] = yaml.safe_load(raw)
    return cfg, args


def resume_checkpoint(value, args):
    """Turn --resume into a concrete last.pt path, so we never guess wrong."""
    if value is True:
        run_dir = resolve(args.get('project', 'runs')) / str(args.get('name', 'exp'))
        ckpt = run_dir / 'weights' / 'last.pt'
    else:
        ckpt = resolve(value)
        if ckpt.is_dir():
            ckpt = ckpt / 'weights' / 'last.pt'

    if not ckpt.is_file():
        found = sorted(str(p) for p in resolve(args.get('project', 'runs')).glob('*/weights/last.pt'))
        die(f'Cannot resume: {ckpt} does not exist.',
            *(['Interrupted runs you can resume:'] + [f'  --resume {f}' for f in found] if found else
              ['No runs/*/weights/last.pt found.']))
    return ckpt


def main():
    cli = parse_args()
    cfg, args = load_config(cli.config)
    cfg, args = apply_overrides(cfg, args, cli)

    data_path, data_cfg = check_dataset(cfg['data'])
    check_cuda(args.get('device', 0))

    # Absolute paths keep runs reproducible no matter which directory you are in.
    args['data'] = str(data_path)
    args['project'] = str(resolve(args.get('project', 'runs')))

    if cli.resume:
        ckpt = resume_checkpoint(cli.resume, args)
        args['resume'] = str(ckpt)
        cfg['model'] = str(ckpt)
        print(f'[info] resuming from {ckpt}')

    # a local checkpoint or architecture file, else a bare name such as
    # yolo11n.pt that Ultralytics downloads on demand
    model_path = resolve(cfg['model'])
    model_ref = str(model_path) if model_path.exists() else str(cfg['model'])

    from ultralytics import YOLO  # imported late so --help stays instant

    print(f'[info] model   : {model_ref}')
    print(f'[info] dataset : {data_path} ({len(data_cfg["names"])} classes)')
    print(f'[info] results : {args["project"]}/{args.get("name", "exp")}')

    model = YOLO(model_ref)
    model.train(**args)


if __name__ == '__main__':
    main()
